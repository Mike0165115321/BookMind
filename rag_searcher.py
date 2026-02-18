"""
RAG Searcher — Hybrid Search (Dense + BM25) with Cross-Encoder reranking.

Search Flow:
  1. Dense Search (FAISS + e5-large) → semantic similarity
  2. BM25 Search (rank-bm25) → keyword matching
  3. Merge scores with configurable weights
  4. Rerank with Cross-Encoder (bge-reranker-v2-m3)
"""
import os
import re
import numpy as np
import torch
import faiss
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import config


def tokenize_thai(text):
    """Simple word-level tokenizer for Thai + English mixed text."""
    tokens = re.findall(r'[\u0E00-\u0E7F]+|[a-zA-Z0-9]+', text.lower())
    return [t for t in tokens if len(t) > 1]


class RAGSearcher:
    def __init__(self, model_embedding=None, model_reranking=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        emb_path = model_embedding or config.MODEL_EMBEDDING
        rerank_path = model_reranking or config.MODEL_RERANKER
        print(f"📡 RAGSearcher using device: {self.device.upper()}")
        self.embedding_model = SentenceTransformer(emb_path, device=self.device)
        self.rerank_model = CrossEncoder(rerank_path, device=self.device)
        self.data = []
        self.index = None
        self.bm25 = None

    def load_index(self, storage_dir=None, index_name=None):
        """Load FAISS index, BM25 corpus, and original text data from disk."""
        storage_dir = storage_dir or config.STORAGE_DIR
        index_name = index_name or config.INDEX_NAME

        # 1. Load FAISS Index
        index_path = os.path.join(storage_dir, f"{index_name}.faiss")
        self.index = faiss.read_index(index_path)

        # 2. Load original text data
        data_path = os.path.join(storage_dir, f"{index_name}_data.pkl")
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        # 3. Load BM25 tokenized corpus
        bm25_path = os.path.join(storage_dir, f"{index_name}_bm25.pkl")
        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                tokenized_corpus = pickle.load(f)
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"✅ Hybrid Search พร้อม! Dense + BM25 ({len(self.data)} chunks)")
        else:
            print(f"⚠️  BM25 data ไม่พบ — ใช้ Dense Search อย่างเดียว ({len(self.data)} chunks)")

    def _dense_search(self, query, top_k):
        """Stage 1a: FAISS retrieval based on semantic similarity."""
        query_emb = self.embedding_model.encode(
            [f"query: {query}"],
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        query_np = query_emb.cpu().detach().numpy().astype('float32')
        scores, indices = self.index.search(query_np, top_k)

        results = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results[idx] = float(score)
        return results

    def _bm25_search(self, query, top_k):
        """Stage 1b: BM25 retrieval based on keyword matching."""
        if self.bm25 is None:
            return {}

        query_tokens = tokenize_thai(query)
        if not query_tokens:
            return {}

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = {}
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results[int(idx)] = float(scores[idx])
        return results

    def _normalize_scores(self, score_dict):
        """Normalize scores to 0-1 range using min-max normalization."""
        if not score_dict:
            return {}
        values = list(score_dict.values())
        min_v, max_v = min(values), max(values)
        if max_v == min_v:
            return {k: 1.0 for k in score_dict}
        return {k: (v - min_v) / (max_v - min_v) for k, v in score_dict.items()}

    def _hybrid_merge(self, dense_scores, bm25_scores):
        """Merge Dense + BM25 scores with configurable weights."""
        dense_norm = self._normalize_scores(dense_scores)
        bm25_norm = self._normalize_scores(bm25_scores)

        # Combine all unique document indices
        all_indices = set(dense_norm.keys()) | set(bm25_norm.keys())

        merged = {}
        for idx in all_indices:
            d_score = dense_norm.get(idx, 0.0)
            b_score = bm25_norm.get(idx, 0.0)
            merged[idx] = (config.HYBRID_DENSE_WEIGHT * d_score +
                           config.HYBRID_BM25_WEIGHT * b_score)

        return merged

    def _should_rerank(self, merged_scores):
        """
        Adaptive Reranking: decide whether to use Cross-Encoder.
        If Top-1 score is clearly dominant (large gap to Top-2), skip reranking.
        Returns: (should_rerank: bool, gap: float)
        """
        if len(merged_scores) < 2:
            return False, 1.0  # Only 1 result, no need to rerank

        sorted_scores = sorted(merged_scores.values(), reverse=True)
        gap = sorted_scores[0] - sorted_scores[1]
        return gap <= config.RERANK_SCORE_GAP, gap

    def search(self, query, top_k=None):
        """
        Full hybrid search pipeline with Adaptive Reranking:
          1. Dense (FAISS) + BM25 retrieval
          2. Merge scores
          3. Check score gap → Rerank only if ambiguous
        """
        top_k = top_k or config.TOP_K_RETRIEVAL

        # Stage 1: Hybrid retrieval
        dense_scores = self._dense_search(query, top_k)
        bm25_scores = self._bm25_search(query, top_k)

        # Stage 2: Merge scores
        if self.bm25 is not None:
            merged = self._hybrid_merge(dense_scores, bm25_scores)
        else:
            merged = dense_scores

        # Get top candidates
        sorted_indices = sorted(merged.keys(), key=lambda x: merged[x], reverse=True)[:top_k]
        retrieved_docs = [self.data[idx] for idx in sorted_indices]
        merged_scores_list = [merged[idx] for idx in sorted_indices]

        if not retrieved_docs:
            return []

        # Stage 3: Adaptive Reranking
        need_rerank, gap = self._should_rerank(merged)

        if need_rerank:
            # Ambiguous → use Cross-Encoder for precision
            print(f"   🔬 Reranking (gap={gap:.3f} ≤ {config.RERANK_SCORE_GAP}) → Cross-Encoder")
            sentence_pairs = [[query, doc] for doc in retrieved_docs]
            rerank_scores = self.rerank_model.predict(sentence_pairs)
            return sorted(zip(retrieved_docs, rerank_scores), key=lambda x: x[1], reverse=True)
        else:
            # Clear winner → skip Reranker, use hybrid scores directly
            print(f"   ⚡ Skip Reranker (gap={gap:.3f} > {config.RERANK_SCORE_GAP}) → Fast mode")
            return list(zip(retrieved_docs, merged_scores_list))