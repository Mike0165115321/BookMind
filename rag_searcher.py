"""
RAG Searcher — Backward compatibility wrapper for the modular Retrieval Pipeline.
"""
from core.retrieval.pipeline import RetrievalPipeline
from core.retrieval.tokenizer import tokenize_thai

class RAGSearcher:
    """
    Thin wrapper around RetrievalPipeline to maintain backward compatibility.
    """
    def __init__(self, model_embedding=None, model_reranking=None):
        # Note: model paths are handled by config in the new modular structure,
        # but we could pass them to the pipeline if needed.
        self.pipeline = RetrievalPipeline()
        # Expose data for backward compatibility if needed
        self.data = self.pipeline.data

    def load_index(self, storage_dir=None, index_name=None):
        self.pipeline.load_index(storage_dir, index_name)
        self.data = self.pipeline.data

    def reload_index(self):
        success = self.pipeline.reload_index()
        self.data = self.pipeline.data
        return success

    def search(self, query, top_k=None):
        return self.pipeline.search(query, top_k)