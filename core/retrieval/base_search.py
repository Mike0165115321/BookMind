"""
Base Search — Dense and BM25 search implementations.
"""
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from core.retrieval.tokenizer import tokenize_thai
import config

class DenseSearcher:
    """Semantic search using FAISS and Sentence Transformers."""
    def __init__(self, model_path: str = None):
        import config
        self.device = "cpu" if getattr(config, "FORCE_CPU_FOR_RAG", False) else ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = model_path or config.MODEL_EMBEDDING
        self.model = SentenceTransformer(model_path, device=self.device)
        self.index = None

    def load_index(self, index_path: str):
        self.index = faiss.read_index(index_path)

    def search(self, query: str, top_k: int) -> dict:
        if self.index is None:
            return {}

        query_emb = self.model.encode(
            [f"query: {query}"],
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        query_np = query_emb.cpu().detach().numpy().astype('float32')
        scores, indices = self.index.search(query_np, top_k)

        results = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results[int(idx)] = float(score)
        return results

class BM25Searcher:
    """Keyword search using Rank-BM25."""
    def __init__(self):
        self.bm25 = None

    def load_corpus(self, tokenized_corpus: list):
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int) -> dict:
        if self.bm25 is None:
            return {}

        query_tokens = tokenize_thai(query)
        if not query_tokens:
            return {}

        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = {}
        for idx in top_indices:
            if scores[idx] > 0:
                results[int(idx)] = float(scores[idx])
        return results

def normalize_scores(score_dict: dict) -> dict:
    """Normalize scores to 0-1 range using min-max normalization."""
    if not score_dict:
        return {}
    values = list(score_dict.values())
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return {k: 1.0 for k in score_dict}
    return {k: (v - min_v) / (max_v - min_v) for k, v in score_dict.items()}

def hybrid_merge(dense_scores: dict, bm25_scores: dict) -> dict:
    """Merge Dense + BM25 scores with configurable weights."""
    dense_norm = normalize_scores(dense_scores)
    bm25_norm = normalize_scores(bm25_scores)

    all_indices = set(dense_norm.keys()) | set(bm25_norm.keys())
    merged = {}
    for idx in all_indices:
        d_score = dense_norm.get(idx, 0.0)
        b_score = bm25_norm.get(idx, 0.0)
        merged[idx] = (config.HYBRID_DENSE_WEIGHT * d_score +
                       config.HYBRID_BM25_WEIGHT * b_score)
    return merged
