"""
Agentic Engine — Pure orchestration logic for Multi-hop RAG.

Handles the execution loop: Decompose -> Search -> Evaluate -> Synthesize.
Emits internal events for observers (like formatters).
"""
import time
from typing import Generator, List, Optional

import config
from rag_searcher import RAGSearcher
from core.query_decomposer import decompose
from core.evaluator import evaluate_sufficiency
from core.agent_memory import AgentMemory
from core.query_transformer import hyde_transform
from core.llm.generator import generate
from core.agentic.types import (
    AgenticResult, 
    InternalEngineEvent, 
    DecompositionResult, 
    EvaluationResult
)
from core.llm.shared.types import ProviderName

class AgenticEngine:
    """
    Core engine for Agentic RAG. 
    Maintains state and orchestrates components.
    """
    def __init__(
        self,
        searcher: RAGSearcher,
        max_iterations: Optional[int] = None,
        sufficiency_threshold: Optional[float] = None,
        max_chunks: Optional[int] = None,
        use_hyde: bool = True,
        provider: str = "gemini",
        model_name: Optional[str] = None,
        hyde_provider: Optional[str] = None,
        hyde_model: Optional[str] = None
    ):
        self.searcher = searcher
        self.max_iterations = max_iterations or config.AGENTIC_MAX_ITERATIONS
        self.sufficiency_threshold = sufficiency_threshold or config.AGENTIC_SUFFICIENCY_THRESHOLD
        self.max_chunks = max_chunks or config.AGENTIC_MAX_CHUNKS
        self.use_hyde = use_hyde and config.ENABLE_HYDE
        self.provider = provider
        self.model_name = model_name
        self.hyde_provider = hyde_provider
        self.hyde_model = hyde_model

    def execute(self, query: str) -> Generator[InternalEngineEvent, None, None]:
        """
        Runs the full agentic loop, yielding internal engine events.
        """
        memory = AgentMemory(original_query=query)
        search_history = []
        
        # 1. Decompose
        decomp = decompose(query)
        yield InternalEngineEvent(
            event_type="decomposed",
            data={
                "query_type": decomp.query_type,
                "sub_queries": decomp.sub_queries,
                "reasoning": decomp.reasoning
            }
        )

        # 2. Search Loop
        pending_queries = list(decomp.sub_queries)
        iteration = 0

        while iteration < self.max_iterations and pending_queries:
            iteration += 1
            
            for sq in pending_queries:
                if memory.has_searched(sq):
                    continue

                yield InternalEngineEvent(
                    event_type="search_started",
                    data={
                        "iteration": iteration,
                        "query": sq,
                        "total_iterations": self.max_iterations
                    }
                )

                search_query = sq
                if self.use_hyde:
                    # Use specific hyde settings if provided, else fallback to generation
                    hp = self.hyde_provider or self.provider
                    hm = self.hyde_model or self.model_name
                    p_enum = ProviderName(hp) if isinstance(hp, str) else hp
                    search_query = hyde_transform(sq, provider=p_enum, model_name=hm)

                results = self.searcher.search(search_query, top_k=config.TOP_K_RETRIEVAL)
                new_count = memory.add_search_results(sq, results, iteration)

                iter_record = {
                    "iteration": iteration,
                    "query": sq,
                    "num_results": len(results),
                    "new_chunks": new_count,
                    "total_chunks": memory.total_chunks,
                }
                search_history.append(iter_record)

                yield InternalEngineEvent(
                    event_type="search_completed",
                    data=iter_record
                )

            # Check limits
            if memory.total_chunks >= self.max_chunks:
                break
            
            # Skip evaluation for simple queries or last iteration
            if decomp.query_type == "simple" or iteration >= self.max_iterations:
                break

            # 3. Evaluate
            eval_result = evaluate_sufficiency(
                original_query=query,
                sub_queries=decomp.sub_queries,
                context_summary=memory.get_context_summary(),
                threshold=self.sufficiency_threshold
            )

            yield InternalEngineEvent(
                event_type="evaluation_completed",
                data={
                    "is_sufficient": eval_result.is_sufficient,
                    "confidence": eval_result.confidence,
                    "missing_aspects": eval_result.missing_aspects,
                    "follow_up_queries": eval_result.follow_up_queries,
                    "reasoning": eval_result.reasoning,
                    "iteration": iteration
                }
            )

            if eval_result.is_sufficient:
                break

            # Prep for next iteration
            pending_queries = [fq for fq in eval_result.follow_up_queries if not memory.has_searched(fq)]
            if not pending_queries:
                break

        # 4. Synthesize
        num_sources = len(decomp.sub_queries)
        display_count = config.TOP_K_DISPLAY * num_sources if num_sources > 1 else config.TOP_K_DISPLAY
        display_chunks = memory.get_balanced_chunks(top_k=display_count)

        yield InternalEngineEvent(
            event_type="synthesis_started",
            data={
                "total_chunks": memory.total_chunks,
                "iterations": iteration,
                "display_chunks": display_chunks
            }
        )

        # Stream the tokens
        p_enum = ProviderName(self.provider) if isinstance(self.provider, str) else self.provider
        for chunk in generate(query, display_chunks, stream=True, provider=p_enum, model_name=self.model_name):
            yield InternalEngineEvent(
                event_type="token",
                data={"text": chunk}
            )

        # 5. Final Result
        result = AgenticResult(
            answer="", # Full answer can be reconstructed from tokens if needed
            sources=display_chunks,
            iterations=iteration,
            query_type=decomp.query_type,
            sub_queries=decomp.sub_queries,
            search_history=search_history,
            total_chunks=memory.total_chunks
        )

        yield InternalEngineEvent(
            event_type="completed",
            data={"result": result}
        )
