"""
Agentic Controller — Orchestrator for Multi-hop RAG with Query Decomposition.

This is the central "brain" that coordinates the entire agentic pipeline:
  1. Decompose the query into sub-queries (QueryDecomposer)
  2. Search for each sub-query (RAGSearcher + optional HyDE)
  3. Evaluate if gathered info is sufficient (Evaluator)
  4. If not sufficient → generate follow-up queries → search again
  5. Repeat until sufficient or max_iterations reached
  6. Synthesize final answer from all gathered chunks (LLM Generator)

Design Principles:
  - Does NOT modify RAGSearcher, LLM Generator, or HyDE — uses them as tools
  - AgentMemory tracks state across iterations
  - Yields events for SSE streaming (web UI progress updates)
  - Falls back to simple pipeline for simple queries (no overhead)
"""
from dataclasses import dataclass, field
from typing import Generator

import config
from rag_searcher import RAGSearcher
from core.query_decomposer import decompose, DecompositionResult
from core.evaluator import evaluate_sufficiency, EvaluationResult
from core.agent_memory import AgentMemory
from core.query_transformer import hyde_transform
from core.llm_generator import generate


# ──────────────────────────────────────────────
# Result Data Structure
# ──────────────────────────────────────────────
@dataclass
class AgenticResult:
    """Final result of the agentic pipeline."""
    answer: str                          # Generated answer
    sources: list[tuple[str, float]]     # All gathered (text, score) pairs
    iterations: int                      # How many search iterations
    query_type: str                      # "simple" or "complex"
    sub_queries: list[str]               # Decomposed sub-queries
    search_history: list[dict]           # Per-iteration search details
    total_chunks: int                    # Total unique chunks gathered


@dataclass
class AgenticEvent:
    """Event emitted during agentic processing (for SSE streaming)."""
    event_type: str    # decompose, search_start, search_done, evaluate, synthesize
    data: dict


class AgenticController:
    """
    Orchestrates the Agentic RAG pipeline.

    Usage:
        controller = AgenticController(searcher)
        result = controller.run(query)
        # or for streaming:
        for event in controller.run_stream(query):
            ...  # handle events
    """

    def __init__(
        self,
        searcher: RAGSearcher,
        max_iterations: int = None,
        sufficiency_threshold: float = None,
        max_chunks: int = None,
        use_hyde: bool = True,
    ):
        self.searcher = searcher
        self.max_iterations = max_iterations or config.AGENTIC_MAX_ITERATIONS
        self.sufficiency_threshold = sufficiency_threshold or config.AGENTIC_SUFFICIENCY_THRESHOLD
        self.max_chunks = max_chunks or config.AGENTIC_MAX_CHUNKS
        self.use_hyde = use_hyde and config.ENABLE_HYDE

    # ──────────────────────────────────────────
    # Main Entry Point (blocking)
    # ──────────────────────────────────────────
    def run(self, query: str, stream_answer: bool = False) -> AgenticResult:
        """
        Run the full agentic pipeline (blocking mode).

        Args:
            query: User's question
            stream_answer: If True, answer field will be a generator

        Returns:
            AgenticResult with answer, sources, and metadata
        """
        events = list(self._execute(query))
        # Last event is the final result
        final = events[-1]
        return final.data["result"]

    # ──────────────────────────────────────────
    # Streaming Entry Point (for Web UI)
    # ──────────────────────────────────────────
    def run_stream(self, query: str) -> Generator[AgenticEvent, None, None]:
        """
        Run the agentic pipeline, yielding events for SSE streaming.

        Each event has an event_type and data dict.
        Event types: decompose, search_start, search_done,
                     evaluate, synthesize, done

        Yields:
            AgenticEvent objects for real-time UI updates
        """
        yield from self._execute(query)

    # ──────────────────────────────────────────
    # Core Pipeline
    # ──────────────────────────────────────────
    def _execute(self, query: str) -> Generator[AgenticEvent, None, None]:
        """
        Internal pipeline execution.

        Flow:
          1. Decompose query
          2. Search each sub-query
          3. Evaluate sufficiency
          4. Loop if needed (with follow-up queries)
          5. Generate final answer
        """
        memory = AgentMemory(original_query=query)
        search_history = []

        # ── Step 1: Decompose ──
        print(f"\n{'═' * 60}")
        print(f"🧠 Agentic RAG: {query}")
        print(f"{'═' * 60}")

        decomp = decompose(query)

        yield AgenticEvent(
            event_type="decompose",
            data={
                "query_type": decomp.query_type,
                "sub_queries": decomp.sub_queries,
                "reasoning": decomp.reasoning,
            },
        )

        # ── Step 2: Search Loop ──
        pending_queries = list(decomp.sub_queries)
        iteration = 0

        while iteration < self.max_iterations and pending_queries:
            iteration += 1
            print(f"\n   ── Iteration {iteration}/{self.max_iterations} ──")

            for sq in pending_queries:
                # Skip if already searched
                if memory.has_searched(sq):
                    print(f"   ⏭️  Skip (already searched): {sq}")
                    continue

                # Notify: starting search
                yield AgenticEvent(
                    event_type="search_start",
                    data={
                        "iteration": iteration,
                        "query": sq,
                        "total_iterations": self.max_iterations,
                    },
                )

                # Optional HyDE transform
                search_query = sq
                if self.use_hyde:
                    search_query = hyde_transform(sq)

                # Execute search
                results = self.searcher.search(
                    search_query,
                    top_k=config.TOP_K_RETRIEVAL,
                )

                # Store in memory (with dedup)
                new_count = memory.add_search_results(sq, results, iteration)

                # Build iteration record
                iter_record = {
                    "iteration": iteration,
                    "query": sq,
                    "num_results": len(results),
                    "new_chunks": new_count,
                    "total_chunks": memory.total_chunks,
                }
                search_history.append(iter_record)

                print(f"   📚 [{sq[:50]}] → {len(results)} results, {new_count} new chunks")

                # Notify: search done
                yield AgenticEvent(
                    event_type="search_done",
                    data=iter_record,
                )

            # Check chunk limit
            if memory.total_chunks >= self.max_chunks:
                print(f"   🛑 Chunk limit reached ({memory.total_chunks}/{self.max_chunks})")
                break

            # ── Step 3: Evaluate (skip for simple queries or last iteration) ──
            if decomp.query_type == "simple" or iteration >= self.max_iterations:
                break

            eval_result = evaluate_sufficiency(
                original_query=query,
                sub_queries=decomp.sub_queries,
                context_summary=memory.get_context_summary(),
                threshold=self.sufficiency_threshold,
            )

            yield AgenticEvent(
                event_type="evaluate",
                data={
                    "is_sufficient": eval_result.is_sufficient,
                    "confidence": eval_result.confidence,
                    "missing_aspects": eval_result.missing_aspects,
                    "follow_up_queries": eval_result.follow_up_queries,
                    "reasoning": eval_result.reasoning,
                    "iteration": iteration,
                },
            )

            if eval_result.is_sufficient:
                print(f"   ✅ Information sufficient! (confidence={eval_result.confidence:.2f})")
                break

            # Prepare follow-up queries for next iteration
            pending_queries = [
                fq for fq in eval_result.follow_up_queries
                if not memory.has_searched(fq)
            ]

            if not pending_queries:
                print(f"   🔄 No new follow-up queries — stopping")
                break

        # ── Step 4: Synthesize Answer ──
        print(f"\n   🤖 Synthesizing from {memory.total_chunks} chunks ({iteration} iterations)")

        # Use balanced selection for complex queries to ensure all sources represented
        num_sources = len(decomp.sub_queries)
        display_count = config.TOP_K_DISPLAY * num_sources if num_sources > 1 else config.TOP_K_DISPLAY
        display_chunks = memory.get_balanced_chunks(top_k=display_count)
        print(f"   📊 Balanced selection: {len(display_chunks)} chunks from {num_sources} sources")

        yield AgenticEvent(
            event_type="synthesize",
            data={
                "total_chunks": memory.total_chunks,
                "iterations": iteration,
                "num_display": len(display_chunks),
            },
        )

        # Generate answer using existing LLM generator
        answer = generate(query, display_chunks, stream=False)

        # Build final result
        result = AgenticResult(
            answer=answer,
            sources=display_chunks,
            iterations=iteration,
            query_type=decomp.query_type,
            sub_queries=decomp.sub_queries,
            search_history=search_history,
            total_chunks=memory.total_chunks,
        )

        yield AgenticEvent(
            event_type="done",
            data={"result": result},
        )

    # ──────────────────────────────────────────
    # Streaming Answer Variant
    # ──────────────────────────────────────────
    def run_stream_with_answer(self, query: str) -> Generator[AgenticEvent, None, None]:
        """
        Like run_stream, but the final answer is streamed token-by-token.

        Instead of a single 'done' event with full answer,
        yields 'token' events followed by 'done'.
        """
        memory = AgentMemory(original_query=query)
        search_history = []

        # ── Step 1: Decompose ──
        print(f"\n{'═' * 60}")
        print(f"🧠 Agentic RAG: {query}")
        print(f"{'═' * 60}")

        decomp = decompose(query)

        yield AgenticEvent(
            event_type="decompose",
            data={
                "query_type": decomp.query_type,
                "sub_queries": decomp.sub_queries,
                "reasoning": decomp.reasoning,
            },
        )

        # ── Step 2: Search Loop (same as _execute) ──
        pending_queries = list(decomp.sub_queries)
        iteration = 0

        while iteration < self.max_iterations and pending_queries:
            iteration += 1
            print(f"\n   ── Iteration {iteration}/{self.max_iterations} ──")

            for sq in pending_queries:
                if memory.has_searched(sq):
                    continue

                yield AgenticEvent(
                    event_type="search_start",
                    data={
                        "iteration": iteration,
                        "query": sq,
                        "total_iterations": self.max_iterations,
                    },
                )

                search_query = sq
                if self.use_hyde:
                    search_query = hyde_transform(sq)

                results = self.searcher.search(
                    search_query,
                    top_k=config.TOP_K_RETRIEVAL,
                )

                new_count = memory.add_search_results(sq, results, iteration)

                iter_record = {
                    "iteration": iteration,
                    "query": sq,
                    "num_results": len(results),
                    "new_chunks": new_count,
                    "total_chunks": memory.total_chunks,
                }
                search_history.append(iter_record)

                print(f"   📚 [{sq[:50]}] → {len(results)} results, {new_count} new chunks")

                yield AgenticEvent(
                    event_type="search_done",
                    data=iter_record,
                )

            if memory.total_chunks >= self.max_chunks:
                break

            if decomp.query_type == "simple" or iteration >= self.max_iterations:
                break

            eval_result = evaluate_sufficiency(
                original_query=query,
                sub_queries=decomp.sub_queries,
                context_summary=memory.get_context_summary(),
                threshold=self.sufficiency_threshold,
            )

            yield AgenticEvent(
                event_type="evaluate",
                data={
                    "is_sufficient": eval_result.is_sufficient,
                    "confidence": eval_result.confidence,
                    "missing_aspects": eval_result.missing_aspects,
                    "follow_up_queries": eval_result.follow_up_queries,
                    "reasoning": eval_result.reasoning,
                    "iteration": iteration,
                },
            )

            if eval_result.is_sufficient:
                break

            pending_queries = [
                fq for fq in eval_result.follow_up_queries
                if not memory.has_searched(fq)
            ]

            if not pending_queries:
                break

        # ── Step 3: Stream Answer (balanced selection) ──
        num_sources = len(decomp.sub_queries)
        display_count = config.TOP_K_DISPLAY * num_sources if num_sources > 1 else config.TOP_K_DISPLAY
        display_chunks = memory.get_balanced_chunks(top_k=display_count)
        print(f"\n   📊 Balanced selection: {len(display_chunks)} chunks from {num_sources} sources")

        # Send sources before generation
        sources_data = []
        for i, (text, score) in enumerate(display_chunks):
            title = text.split("]")[0].lstrip("[") if "[" in text else "ไม่ระบุ"
            sources_data.append({
                "rank": i + 1,
                "title": title,
                "text": text[:300],
                "score": round(float(score), 3),
            })

        yield AgenticEvent(
            event_type="sources",
            data={
                "sources": sources_data,
                "total_chunks": memory.total_chunks,
                "iterations": iteration,
            },
        )

        yield AgenticEvent(
            event_type="synthesize",
            data={
                "total_chunks": memory.total_chunks,
                "iterations": iteration,
            },
        )

        # Stream tokens
        for chunk in generate(query, display_chunks, stream=True):
            yield AgenticEvent(
                event_type="token",
                data={"text": chunk},
            )

        # Done
        yield AgenticEvent(
            event_type="done",
            data={
                "iterations": iteration,
                "query_type": decomp.query_type,
                "sub_queries": decomp.sub_queries,
                "total_chunks": memory.total_chunks,
                "search_history": search_history,
            },
        )
"""
"""
