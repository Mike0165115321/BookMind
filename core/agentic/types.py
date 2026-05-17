"""
Agentic Types — Shared data structures for the Agentic RAG system.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class AgenticResult:
    """Final output of the agentic process."""
    answer: str
    sources: List[Any]
    iterations: int
    query_type: str
    sub_queries: List[str]
    search_history: List[Dict]
    total_chunks: int

@dataclass
class InternalEngineEvent:
    """Event emitted by the engine during processing."""
    event_type: str  # e.g., "decomposed", "search_started", "search_completed", "evaluation_completed", "synthesis_started", "token", "completed"
    data: Dict[str, Any]

@dataclass
class DecompositionResult:
    """Result from QueryDecomposer."""
    query_type: str
    sub_queries: List[str]
    reasoning: str = ""
    original_query: str = ""
    context_budget: int = 5

@dataclass
class EvaluationResult:
    """Result from Evaluator."""
    is_sufficient: bool
    confidence: float
    missing_aspects: List[str]
    follow_up_queries: List[str]
    reasoning: str = ""

@dataclass
class AgenticEvent:
    """Event formatted for UI (SSE)."""
    event_type: str
    data: Dict[str, Any]
