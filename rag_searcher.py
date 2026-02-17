"""
RAG Searcher — Loads FAISS index and performs two-stage search
(Bi-Encoder retrieval + Cross-Encoder reranking).
"""
import os
import torch
import faiss
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
import config


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

    def load_index(self, storage_dir=None, index_name=None):
        """Load FAISS index and original text data from disk."""
        storage_dir = storage_dir or config.STORAGE_DIR
        index_name = index_name or config.INDEX_NAME

        # 1. Load FAISS Index
        index_path = os.path.join(storage_dir, f"{index_name}.faiss")
        self.index = faiss.read_index(index_path)

        # 2. Load original text data
        data_path = os.path.join(storage_dir, f"{index_name}_data.pkl")
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        print(f"✅ โหลด Index และข้อมูล {len(self.data)} รายการ พร้อมค้นหาค่ะ")

    def search(self, query, top_k=None):
        """Two-stage search: FAISS retrieval + Cross-Encoder reranking."""
        top_k = top_k or config.TOP_K_RETRIEVAL

        # 1. Query Embedding (Bi-Encoder)
        query_emb = self.embedding_model.encode(
            [f"query: {query}"],
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        query_np = query_emb.cpu().detach().numpy().astype('float32')

        # 2. FAISS retrieval
        scores, indices = self.index.search(query_np, top_k)
        retrieved_docs = [self.data[idx] for idx in indices[0] if idx != -1]

        if not retrieved_docs:
            return []

        # 3. Reranking (Cross-Encoder)
        sentence_pairs = [[query, doc] for doc in retrieved_docs]
        rerank_scores = self.rerank_model.predict(sentence_pairs)

        return sorted(zip(retrieved_docs, rerank_scores), key=lambda x: x[1], reverse=True)