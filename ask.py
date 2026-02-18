"""
Entry Point — Full RAG Pipeline (Query Transform → Search → Generate).

Pipeline:
  1. [Optional] HyDE Query Transform (Groq LLaMA) — improves search quality
  2. Hybrid Search (Dense + BM25) + Adaptive Reranking
  3. LLM Generation (Gemini 2.5 Flash) — answers from retrieved context

Usage:
  python3 ask.py                     # Interactive mode
  python3 ask.py "your question"     # Single question mode
  python3 ask.py --no-hyde "q"       # Disable HyDE
  python3 ask.py --no-stream "q"     # Disable streaming
"""
import sys
import time
import config
from rag_searcher import RAGSearcher
from core.llm_generator import generate
from core.query_transformer import hyde_transform


def ask(query: str, searcher: RAGSearcher, stream: bool = True, use_hyde: bool = True) -> str:
    """
    Full RAG pipeline: [HyDE] → Search → Generate.

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


def main():
    """Main entry point with interactive and single-question modes."""
    stream = "--no-stream" not in sys.argv
    use_hyde = "--no-hyde" not in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    # Initialize searcher
    print("=" * 60)
    print("🤖 RAG System — Full Pipeline")
    print(f"   📡 Search: Dense + BM25 + Adaptive Reranking")
    print(f"   🪄 HyDE: {'ON' if use_hyde and config.ENABLE_HYDE else 'OFF'} ({config.GROQ_MODEL})")
    print(f"   🧠 LLM:  {config.GEMINI_MODEL}")
    print("=" * 60)
    searcher = RAGSearcher()
    searcher.load_index()

    if args:
        # Single question mode
        ask(" ".join(args), searcher, stream=stream, use_hyde=use_hyde)
    else:
        # Interactive mode
        print("\n💬 Interactive mode — type your question (or 'q' to quit)\n")
        while True:
            try:
                query = input("❓ คำถาม: ").strip()
                if not query or query.lower() in ("q", "quit", "exit"):
                    print("👋 ออกจากระบบ")
                    break
                ask(query, searcher, stream=stream, use_hyde=use_hyde)
                print()
            except KeyboardInterrupt:
                print("\n👋 ออกจากระบบ")
                break


if __name__ == "__main__":
    main()
