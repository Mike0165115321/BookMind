"""
Reranker — Semantic reranking using Cross-Encoder models.
"""
import torch
from sentence_transformers import CrossEncoder
import config

class Reranker:
    """
    Handles re-scoring and re-ordering of retrieved documents using a Cross-Encoder.
    Supports adaptive reranking based on score gaps.
    """
    def __init__(self, model_path: str = None):
        import config
        self.device = "cpu" if getattr(config, "FORCE_CPU_FOR_RAG", False) else ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = model_path or config.MODEL_RERANKER
        self.model = CrossEncoder(model_path, device=self.device)

    def should_rerank(self, merged_scores: dict) -> tuple[bool, float]:
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

    def rerank(self, query: str, retrieved_docs: list, initial_scores: list) -> list:
        """
        Rerank a list of documents using the Cross-Encoder model.
        """
        if not retrieved_docs:
            return []

        sentence_pairs = [[query, doc] for doc in retrieved_docs]
        rerank_scores = self.model.predict(sentence_pairs)
        
        results = sorted(zip(retrieved_docs, rerank_scores), key=lambda x: x[1], reverse=True)
        
        # Filter by threshold
        return [r for r in results if r[1] >= config.RELEVANCE_THRESHOLD]
