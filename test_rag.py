
import argparse
import os
import time
import config
from rag_searcher import RAGSearcher

# ──────────────────────────────────────────────
# Predefined test queries (add/remove as needed)
# ──────────────────────────────────────────────
TEST_QUERIES = [
    "Atomic Habits สอนอะไรเกี่ยวกับการสร้างนิสัย",
    "Rich Dad Poor Dad แนะนำอะไรเกี่ยวกับการเงิน",
    "วิธีเจรจาต่อรองที่ดี",
    "วิธีฝึกสมาธิและจัดการอารมณ์",
    "ซุนวูสอนอะไรเกี่ยวกับกลยุทธ์",
]


def run_test(searcher, query, top_k):
    """Run a single query and print results with timing."""
    print(f"\n{'─' * 60}")
    print(f"🔎 Query: {query}")
    print(f"{'─' * 60}")

    start = time.time()
    results = searcher.search(query, top_k=config.TOP_K_RETRIEVAL)
    elapsed = time.time() - start

    if not results:
        print("   ❌ ไม่พบข้อมูลที่เกี่ยวข้อง")
        return

    for i, (doc, score) in enumerate(results[:top_k]):
        print(f"  [{i+1}] Score: {score:.4f}")
        print(f"      {doc[:200]}...")

    print(f"  ⏱️ {elapsed:.3f}s")


def main():
    parser = argparse.ArgumentParser(description="Test RAG search quality")
    parser.add_argument("--query", type=str, help="Single query to test")
    parser.add_argument("--top_k", type=int, default=config.TOP_K_DISPLAY, help="Results per query")
    args = parser.parse_args()

    # Verify index
    index_path = os.path.join(config.STORAGE_DIR, f"{config.INDEX_NAME}.faiss")
    if not os.path.exists(index_path):
        print("❌ Index not found! Please build it first:")
        print("   python3 build_index.py")
        return

    # Load searcher
    searcher = RAGSearcher()
    searcher.load_index(storage_dir=config.STORAGE_DIR, index_name=config.INDEX_NAME)

    print(f"\n{'=' * 60}")
    print(f"🧪 RAG Test Suite — {config.INDEX_NAME}")
    print(f"{'=' * 60}")

    if args.query:
        # Single query mode
        run_test(searcher, args.query, args.top_k)
    else:
        # Run all predefined queries
        for query in TEST_QUERIES:
            run_test(searcher, query, args.top_k)

    print(f"\n{'=' * 60}")
    print(f"✅ Testing complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
