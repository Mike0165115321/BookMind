import time
import logging
from typing import Generator, List, Dict, Any, Optional

import config
from rag_searcher import RAGSearcher
from core.query_decomposer import decompose
from core.evaluator import evaluate_sufficiency
from core.agent_memory import AgentMemory
from core.query_transformer import hyde_transform
from core.llm.shared.types import ProviderName
from core.agentic.types import (
    AgenticResult, 
    InternalEngineEvent, 
    DecompositionResult, 
    EvaluationResult
)

logger = logging.getLogger(__name__)

class AgenticEngine:
    """
    Core engine for Agentic RAG. 
    Maintains state and orchestrates components.
    """
    def __init__(
        self, 
        searcher: RAGSearcher,
        use_hyde: bool = True,
        max_iterations: int = 3,
        max_chunks: int = 15,
        sufficiency_threshold: float = 0.8,
        provider: ProviderName = ProviderName.GEMINI,
        model_name: str = None,
        hyde_provider: str = None,
        hyde_model: str = None,
        agentic_provider: str = None,
        agentic_model: str = None,
        persona_id: str = "default"
    ):
        self.searcher = searcher
        self.use_hyde = use_hyde
        self.max_iterations = max_iterations
        self.max_chunks = max_chunks
        self.sufficiency_threshold = sufficiency_threshold
        self.provider = provider
        self.model_name = model_name
        self.hyde_provider = hyde_provider
        self.hyde_model = hyde_model
        
        # Settings for Decomposer & Evaluator
        self.agentic_provider = agentic_provider
        self.agentic_model = agentic_model
        self.persona_id = persona_id

    def execute(self, query: str) -> Generator[InternalEngineEvent, None, None]:
        """
        Runs the full agentic loop, yielding internal engine events.
        """
        memory = AgentMemory(original_query=query)
        search_history = []
        
        # 1. Decompose
        decomp = decompose(
            query=query, 
            agentic_provider=self.agentic_provider, 
            agentic_model=self.agentic_model
        )
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

                # Agentic loop uses raw sub-queries for retrieval (No HyDE to save cost/time)
                results = self.searcher.search(sq, top_k=config.TOP_K_RETRIEVAL, context_budget=decomp.context_budget)
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

            if memory.total_chunks >= self.max_chunks:
                break
            
            if decomp.query_type == "simple" or iteration >= self.max_iterations:
                break

            # 3. Evaluate
            eval_result = evaluate_sufficiency(
                query=query,
                context=memory.get_context_summary(),
                agentic_provider=self.agentic_provider,
                agentic_model=self.agentic_model
            )

            yield InternalEngineEvent(
                event_type="evaluation_completed",
                data={
                    "is_sufficient": eval_result.is_sufficient,
                    "confidence": eval_result.confidence,
                    "missing_aspects": eval_result.missing_aspects,
                    "follow_up_queries": eval_result.follow_up_queries,
                    "reasoning": eval_result.reasoning
                }
            )

            if eval_result.is_sufficient:
                break
            
            pending_queries = eval_result.follow_up_queries

        # 4. Final Synthesis
        yield InternalEngineEvent(
            event_type="synthesis_started",
            data={
                "total_chunks": memory.total_chunks,
                "iterations": iteration,
                "display_chunks": memory.get_all_chunks()[:config.TOP_K_DISPLAY]
            }
        )

        from core.llm.generator import generate
        full_answer = ""
        
        # Use main provider/model for synthesis
        for chunk in generate(
            query=query, 
            search_results=memory.get_all_chunks(), 
            stream=True, 
            provider=self.provider, 
            model_name=self.model_name,
            persona_id=self.persona_id
        ):
            full_answer += chunk.text if hasattr(chunk, 'text') else str(chunk)
            yield InternalEngineEvent(event_type="token", data={"text": chunk})

        result = AgenticResult(
            answer=full_answer,
            sources=memory.get_all_chunks(),
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
