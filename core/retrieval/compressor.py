import re
# pyrefly: ignore [missing-import]
import torch
import config
from typing import List, Tuple

class SentenceSplitter:
    """Splits a chunk of text into sentences."""
    @staticmethod
    def split(text: str) -> List[str]:
        # Clean text
        text = text.strip()
        
        # Split by newlines, bullet points, slashes, or end of sentence punctuation
        # Thai text often uses spaces or newlines for sentence boundaries in chunks.
        # This regex handles basic splitting cases.
        pattern = r'[\n\r]+|(?<=[.!?])\s+|\s+/\s+|\s+-\s+'
        sentences = re.split(pattern, text)
        
        # Filter out very short sentences
        valid_sentences = []
        for s in sentences:
            s = s.strip()
            if len(s) >= config.COMPRESSION_MIN_SENTENCE_LENGTH:
                valid_sentences.append(s)
                
        return valid_sentences

class EmbeddingFilter:
    """Filters sentences using Cosine Similarity with the query."""
    def __init__(self, model):
        self.model = model  # SentenceTransformer

    def filter(self, query: str, sentences: List[str]) -> List[Tuple[str, float]]:
        if not sentences:
            return []
            
        # Encode query
        query_emb = self.model.encode(
            [f"query: {query}"],
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        # Encode sentences (pass as documents)
        sent_embs = self.model.encode(
            [f"passage: {s}" for s in sentences],
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        # Compute cosine similarities (dot product since normalized)
        cos_scores = torch.nn.functional.cosine_similarity(query_emb, sent_embs)
        
        # Filter based on threshold
        filtered = []
        for i, score in enumerate(cos_scores):
            score_val = score.item()
            if score_val >= config.COMPRESSION_EMBEDDING_THRESHOLD:
                filtered.append((sentences[i], score_val))
                
        return filtered

class SentenceReranker:
    """Reranks sentences using Cross-Encoder."""
    def __init__(self, model):
        self.model = model  # CrossEncoder
        
    def rerank(self, query: str, sentences: List[str]) -> List[Tuple[str, float]]:
        if not sentences:
            return []
            
        pairs = [[query, s] for s in sentences]
        scores = self.model.predict(pairs)
        
        results = [(s, float(score)) for s, score in zip(sentences, scores)]
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

class SentenceCompressor:
    """Orchestrates sentence-level context compression."""
    def __init__(self, dense_model, reranker_model):
        self.splitter = SentenceSplitter()
        self.filter = EmbeddingFilter(dense_model)
        self.reranker = SentenceReranker(reranker_model)
        
    def compress(self, query: str, retrieved_docs: List[str], top_n: int) -> List[Tuple[str, float]]:
        """
        Compresses chunks into top-N most relevant sentences.
        Expects retrieved_docs to be a list of strings (or strings with metadata).
        """
        all_sentences = []
        doc_names = []
        
        # Step 1: Split all chunks into sentences
        for doc in retrieved_docs:
            doc_name = "ไม่ระบุ"
            text_to_split = doc
            
            # Extract doc name if present e.g. "[Filename.pdf] Content..."
            if doc.startswith("[") and "]" in doc:
                parts = doc.split("]", 1)
                doc_name = parts[0].lstrip("[")
                text_to_split = parts[1]
                
            sentences = self.splitter.split(text_to_split)
            for s in sentences:
                all_sentences.append(s)
                doc_names.append(doc_name)
                
        if not all_sentences:
            return []
            
        # Step 2: Embedding Filter (Coarse)
        filtered_results = self.filter.filter(query, all_sentences)
        
        if not filtered_results:
            return []
            
        # We need to map back to doc names. We'll reconstruct the full string
        filtered_sentences = [r[0] for r in filtered_results]
        
        # Step 3: Reranker Score (Fine)
        reranked_results = self.reranker.rerank(query, filtered_sentences)
        
        # Take Top-N
        top_results = reranked_results[:top_n]
        
        # Reconstruct "[DocName] Context Window" format
        final_results = []
        for sent, score in top_results:
            # Find the original doc_name for this sentence
            idx = all_sentences.index(sent)
            d_name = doc_names[idx]
            
            # Extract window (-1 to +1) to provide context
            start_idx = max(0, idx - 1)
            end_idx = min(len(all_sentences) - 1, idx + 1)
            
            window_sentences = []
            for j in range(start_idx, end_idx + 1):
                if doc_names[j] == d_name:
                    window_sentences.append(all_sentences[j])
                    
            context_window = " ".join(window_sentences)
            formatted_text = f"[{d_name}] {context_window}"
            
            # Avoid exact duplicates if windows overlap
            if not any(context_window in r[0] for r in final_results):
                final_results.append((formatted_text, score))
            
        return final_results
