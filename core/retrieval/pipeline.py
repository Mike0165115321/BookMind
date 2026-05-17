"""
Retrieval Pipeline — Orchestrates the full hybrid search and reranking flow.
"""
import os
import pickle
import config
from core.retrieval.base_search import DenseSearcher, BM25Searcher, hybrid_merge
from core.retrieval.reranker import Reranker

class RetrievalPipeline:
    """
    Coordinates the retrieval process:
    1. Hybrid Search (Dense + BM25)
    2. Score Merging
    3. Adaptive Reranking
    """
    def __init__(self):
        self.dense_searcher = DenseSearcher()
        self.bm25_searcher = BM25Searcher()
        self.reranker = Reranker()
        self.data = []
        
        if getattr(config, 'COMPRESSION_ENABLED', False):
            from core.retrieval.compressor import SentenceCompressor
            self.compressor = SentenceCompressor(self.dense_searcher.model, self.reranker.model)
        else:
            self.compressor = None

    def load_index(self, storage_dir: str = None, index_name: str = None):
        """Load all index components from disk."""
        storage_dir = storage_dir or config.STORAGE_DIR
        index_name = index_name or config.INDEX_NAME

        # 1. FAISS
        index_path = os.path.join(storage_dir, f"{index_name}.faiss")
        self.dense_searcher.load_index(index_path)

        # 2. Original Data
        data_path = os.path.join(storage_dir, f"{index_name}_data.pkl")
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        # 3. BM25
        bm25_path = os.path.join(storage_dir, f"{index_name}_bm25.pkl")
        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                tokenized_corpus = pickle.load(f)
            self.bm25_searcher.load_corpus(tokenized_corpus)
            print(f"✅ Hybrid Search Ready! ({len(self.data)} chunks)")
        else:
            print(f"⚠️ BM25 data missing — using Dense Search only.")

    def reload_index(self):
        """Reload all components from disk."""
        print("🔄 Hot-reloading Retrieval Index...")
        try:
            self.load_index()
            return True
        except Exception as e:
            print(f"❌ Reload failed: {e}")
            return False

    def search(self, query: str, top_k: int = None, context_budget: int = None):
        """Execute the full retrieval pipeline."""
        top_k = top_k or config.TOP_K_RETRIEVAL

        # Stage 1: Hybrid Retrieval
        dense_scores = self.dense_searcher.search(query, top_k)
        bm25_scores = self.bm25_searcher.search(query, top_k)

        # Stage 2: Merge
        merged = hybrid_merge(dense_scores, bm25_scores)
        
        # Sort and get candidates
        sorted_indices = sorted(merged.keys(), key=lambda x: merged[x], reverse=True)[:top_k]
        retrieved_docs = [self.data[idx] for idx in sorted_indices]
        merged_scores_list = [merged[idx] for idx in sorted_indices]

        if not retrieved_docs:
            return []

        # Stage 3: Adaptive Reranking
        need_rerank, gap = self.reranker.should_rerank(merged)

        if need_rerank:
            print(f"   🔬 Reranking (gap={gap:.3f} ≤ {config.RERANK_SCORE_GAP}) → Cross-Encoder")
            final_results = self.reranker.rerank(query, retrieved_docs, merged_scores_list)
        else:
            print(f"   ⚡ Skip Reranker (gap={gap:.3f} > {config.RERANK_SCORE_GAP}) → Fast mode")
            results = list(zip(retrieved_docs, merged_scores_list))
            final_results = [r for r in results if r[1] >= config.RELEVANCE_THRESHOLD]
            
        if getattr(config, 'COMPRESSION_ENABLED', False) and self.compressor:
            budget = context_budget or getattr(config, 'COMPRESSION_TOP_N_SIMPLE', 5)
            doc_texts = [r[0] for r in final_results]
            print(f"   ✂️  Compressing {len(doc_texts)} chunks into Top-{budget} sentences")
            return self.compressor.compress(query, doc_texts, budget)
            
        return final_results
