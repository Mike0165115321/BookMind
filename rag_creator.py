"""
RAG Creator — Builds FAISS index + BM25 corpus from source documents.
"""
import os
import re
import json
import torch
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import config
from core.document_loader import DocumentLoader
from core.retrieval.tokenizer import tokenize_thai

class TextChunker:
    """Splits long text into overlapping chunks at natural boundaries."""
    def __init__(self, chunk_size=None, chunk_overlap=None):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @property
    def chunk_size(self):
        return self._chunk_size or config.CHUNK_SIZE

    @property
    def chunk_overlap(self):
        return self._chunk_overlap or config.CHUNK_OVERLAP

    def chunk(self, text, metadata_prefix=""):
        if not text.strip(): return []
        
        # 1. Normalize and split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks = []
        current_chunk_text = ""

        for para in paragraphs:
            # Handle exceptionally long paragraphs by splitting them into sub-paragraphs with overlap
            sub_paras = []
            if len(para) > self.chunk_size:
                temp_p = para
                while len(temp_p) > self.chunk_size:
                    # Take a slice of chunk_size
                    sub_paras.append(temp_p[:self.chunk_size])
                    # Move forward but keep the overlap
                    temp_p = temp_p[self.chunk_size - self.chunk_overlap:]
                if temp_p:
                    sub_paras.append(temp_p)
            else:
                sub_paras = [para]
                
            # Process each sub-paragraph (or the original paragraph)
            for p in sub_paras:
                if not current_chunk_text:
                    current_chunk_text = p
                elif len(current_chunk_text) + len(p) + 1 <= self.chunk_size:
                    current_chunk_text += "\n" + p
                else:
                    # Current chunk is full, save it
                    full_text = f"{metadata_prefix}\n{current_chunk_text}".strip() if metadata_prefix else current_chunk_text
                    chunks.append(full_text)
                    
                    # Start new chunk with overlap from the END of the previous chunk text
                    overlap_text = current_chunk_text[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                    current_chunk_text = (overlap_text + "\n" + p).strip() if overlap_text else p
                    
        # Don't forget the last buffer
        if current_chunk_text:
            full_text = f"{metadata_prefix}\n{current_chunk_text}".strip() if metadata_prefix else current_chunk_text
            chunks.append(full_text)
            
        return chunks

class RAGCreator:
    def __init__(self, model_name=None):
        import config
        self.device = "cpu" if getattr(config, "FORCE_CPU_FOR_RAG", False) else ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = model_name or config.MODEL_EMBEDDING
        self.model = SentenceTransformer(model_path, device=self.device)
        self.chunker = TextChunker()
        self.data = []

    def process_single_file(self, filepath, book_title=None, category=None):
        """Processes a file using the DocumentLoader and returns chunks."""
        loaded_docs = DocumentLoader.load(filepath, book_title)
        all_chunks = []
        for doc in loaded_docs:
            # Use category if provided and not already in prefix
            prefix = doc["metadata_prefix"]
            if category and category not in prefix:
                prefix = f"[{category}] {prefix}".strip()
                
            chunks = self.chunker.chunk(doc["content"], metadata_prefix=prefix)
            all_chunks.extend(chunks)
        return all_chunks

    def create_embeddings(self, docs):
        """Generate embeddings for a list of documents."""
        if not docs: return None
        prefixed_docs = [f"passage: {doc}" for doc in docs]
        embeddings = self.model.encode(
            prefixed_docs,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=config.BATCH_SIZE
        )
        return embeddings.cpu().detach().numpy().astype('float32')

    def update_index(self, new_docs):
        """Incremental update for the index on disk."""
        if not new_docs: return False
        
        index_path = os.path.join(config.STORAGE_DIR, f"{config.INDEX_NAME}.faiss")
        data_path = os.path.join(config.STORAGE_DIR, f"{config.INDEX_NAME}_data.pkl")
        bm25_path = os.path.join(config.STORAGE_DIR, f"{config.INDEX_NAME}_bm25.pkl")
        
        # Load existing
        if not os.path.exists(index_path):
            self.data = new_docs
            return self.build_and_save()
            
        index = faiss.read_index(index_path)
        with open(data_path, "rb") as f: existing_data = pickle.load(f)
        with open(bm25_path, "rb") as f: existing_bm25 = pickle.load(f)
        
        # Add new
        new_embs = self.create_embeddings(new_docs)
        index.add(new_embs)
        existing_data.extend(new_docs)
        existing_bm25.extend([tokenize_thai(d) for d in new_docs])
        
        # Save
        faiss.write_index(index, index_path)
        with open(data_path, "wb") as f: pickle.dump(existing_data, f)
        with open(bm25_path, "wb") as f: pickle.dump(existing_bm25, f)
        return True

    def build_and_save(self, source_path=None):
        """Full build from directory."""
        source_path = source_path or config.DATA_DIR
        if not self.data:
            for root, dirs, files in os.walk(source_path):
                for filename in sorted(files):
                    filepath = os.path.join(root, filename)
                    if os.path.isfile(filepath):
                        self.data.extend(self.process_single_file(filepath))
        
        if not self.data: return False
        
        embeddings = self.create_embeddings(self.data)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        tokenized_corpus = [tokenize_thai(d) for d in self.data]
        
        os.makedirs(config.STORAGE_DIR, exist_ok=True)
        faiss.write_index(index, os.path.join(config.STORAGE_DIR, f"{config.INDEX_NAME}.faiss"))
        with open(os.path.join(config.STORAGE_DIR, f"{config.INDEX_NAME}_data.pkl"), "wb") as f:
            pickle.dump(self.data, f)
        with open(os.path.join(config.STORAGE_DIR, f"{config.INDEX_NAME}_bm25.pkl"), "wb") as f:
            pickle.dump(tokenized_corpus, f)
        return True

    def _tokenize(self, text):
        tokens = re.findall(r'[\u0E00-\u0E7F]+|[a-zA-Z0-9]+', text.lower())
        return [t for t in tokens if len(t) > 1]