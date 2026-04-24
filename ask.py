"""
Entry Point — Full RAG Pipeline (Query Transform → Search → Generate).

Supports two modes:
  - Classic:  [HyDE] → Search → Generate (single-shot)
  - Agentic:  Decompose → Multi-hop Search → Evaluate → Generate (multi-hop)

Pipeline (Classic):
  1. [Optional] HyDE Query Transform (Groq LLaMA) — improves search quality
  2. Hybrid Search (Dense + BM25) + Adaptive Reranking
  3. LLM Generation (Gemini 2.5 Flash) — answers from retrieved context

Pipeline (Agentic):
  1. Query Decomposition — break complex query into sub-queries
  2. Multi-hop Search — search each sub-query, evaluate, loop if needed
  3. LLM Generation — synthesize answer from all gathered chunks

Usage:
  python3 ask.py                         # Interactive mode (classic)
  python3 ask.py "your question"         # Single question (classic)
  python3 ask.py --agentic "question"    # Agentic mode (multi-hop)
  python3 ask.py --no-hyde "q"           # Disable HyDE
  python3 ask.py --no-stream "q"         # Disable streaming
"""
import sys
import time
import config
from rag_searcher import RAGSearcher
from core.llm_generator import generate
from core.query_transformer import hyde_transform
from core.agentic_controller import AgenticController


def ask(query: str, searcher: RAGSearcher, stream: bool = True, use_hyde: bool = True) -> str:
    """
    Classic RAG pipeline: [HyDE] → Search → Generate.

    Args:
        query: User's question
        searcher: Initialized RAGSearcher instance
        stream: Whether to stream the response
        use_hyde: Whether to use HyDE query transform

    Returns:
        Generated answer string
    """
    print(f"\n{'═' * 60}")
    print(f"❓ {query}")
    print(f"{'═' * 60}")

    # Stage 0: Query Transform (HyDE via Groq)
    search_query = query
    if use_hyde and config.ENABLE_HYDE:
        t_hyde = time.time()
        search_query = hyde_transform(query)
        hyde_time = time.time() - t_hyde
    else:
        hyde_time = 0

    # Stage 1+2: Retrieval (Hybrid Search + Adaptive Reranking)
    t0 = time.time()
    results = searcher.search(search_query, top_k=config.TOP_K_RETRIEVAL)
    search_time = time.time() - t0
    print(f"   📚 {len(results)} chunks retrieved ({search_time:.3f}s)")

    # Show top sources
    for i, (text, score) in enumerate(results[:config.TOP_K_DISPLAY], 1):
        title = text.split("]")[0].lstrip("[") if "[" in text else "—"
        snippet = text[:80].replace("\n", " ")
        print(f"   [{i}] ({score:.2f}) [{title}] {snippet}...")

    # Stage 3: Generation (Gemini LLM) — use ORIGINAL query, not HyDE
    print(f"\n   🤖 Generating with {config.GEMINI_MODEL}...")
    t1 = time.time()

    if stream:
        print(f"\n{'─' * 60}")
        full_response = ""
        for chunk in generate(query, results[:config.TOP_K_DISPLAY], stream=True):
            print(chunk, end="", flush=True)
            full_response += chunk
        print(f"\n{'─' * 60}")
    else:
        full_response = generate(query, results[:config.TOP_K_DISPLAY], stream=False)
        print(f"\n{'─' * 60}")
        print(full_response)
        print(f"{'─' * 60}")

    gen_time = time.time() - t1
    total = hyde_time + search_time + gen_time

    # Timing summary
    parts = []
    if hyde_time > 0:
        parts.append(f"HyDE: {hyde_time:.2f}s")
    parts.append(f"Search: {search_time:.3f}s")
    parts.append(f"Generate: {gen_time:.2f}s")
    parts.append(f"Total: {total:.2f}s")
    print(f"   ⏱️  {' | '.join(parts)}")

    return full_response


def ask_agentic(query: str, searcher: RAGSearcher, stream: bool = True, use_hyde: bool = True) -> str:
    """
    Agentic RAG pipeline: Decompose → Multi-hop Search → Evaluate → Generate.

    Uses AgenticController to:
      1. Decompose complex queries into sub-queries
      2. Search each sub-query with optional HyDE
      3. Evaluate if information is sufficient
      4. Loop with follow-up queries if needed
      5. Synthesize final answer from all gathered chunks

    Args:
        query: User's question
        searcher: Initialized RAGSearcher instance
        stream: Whether to stream the response
        use_hyde: Whether to use HyDE query transform

    Returns:
        Generated answer string
    """
    t_total = time.time()

    controller = AgenticController(searcher=searcher, use_hyde=use_hyde)

    if stream:
        # Stream mode: print events + stream answer tokens
        full_response = ""

        for event in controller.run_stream_with_answer(query):
            if event.event_type == "token":
                if not full_response:
                    # First token — print header
                    print(f"\n{'─' * 60}")
                print(event.data["text"], end="", flush=True)
                full_response += event.data["text"]

            elif event.event_type == "done":
                print(f"\n{'─' * 60}")
                total_time = time.time() - t_total
                d = event.data
                print(f"   ⏱️  Total: {total_time:.2f}s | "
                      f"Iterations: {d['iterations']} | "
                      f"Chunks: {d['total_chunks']} | "
                      f"Type: {d['query_type']}")

        return full_response

    else:
        # Blocking mode
        result = controller.run(query)

        print(f"\n{'─' * 60}")
        print(result.answer)
        print(f"{'─' * 60}")

        total_time = time.time() - t_total
        print(f"   ⏱️  Total: {total_time:.2f}s | "
              f"Iterations: {result.iterations} | "
              f"Chunks: {result.total_chunks} | "
              f"Type: {result.query_type}")

        return result.answer


def main():
    """Main entry point with interactive and single-question modes."""
    stream = "--no-stream" not in sys.argv
    use_hyde = "--no-hyde" not in sys.argv
    agentic = "--agentic" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    # Determine mode
    mode_name = "Agentic" if agentic else "Classic"
    ask_fn = ask_agentic if agentic else ask

    # Initialize searcher
    print("=" * 60)
    print(f"🤖 RAG System — {mode_name} Pipeline")
    print(f"   📡 Search: Dense + BM25 + Adaptive Reranking")
    print(f"   🪄 HyDE: {'ON' if use_hyde and config.ENABLE_HYDE else 'OFF'} ({config.GROQ_MODEL})")
    print(f"   🧠 LLM:  {config.GEMINI_MODEL}")
    if agentic:
        print(f"   🔄 Agentic: ON (max {config.AGENTIC_MAX_ITERATIONS} iterations, "
              f"threshold {config.AGENTIC_SUFFICIENCY_THRESHOLD})")
    print("=" * 60)
    searcher = RAGSearcher()
    searcher.load_index()

    if args:
        # Single question mode
        ask_fn(" ".join(args), searcher, stream=stream, use_hyde=use_hyde)
    else:
        # Interactive mode
        print(f"\n💬 Interactive mode ({mode_name}) — type your question (or 'q' to quit)\n")
        while True:
            try:
                query = input("❓ คำถาม: ").strip()
                if not query or query.lower() in ("q", "quit", "exit"):
                    print("👋 ออกจากระบบ")
                    break
                ask_fn(query, searcher, stream=stream, use_hyde=use_hyde)
                print()
            except KeyboardInterrupt:
                print("\n👋 ออกจากระบบ")
                break


if __name__ == "__main__":
    main()
