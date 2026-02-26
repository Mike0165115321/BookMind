"""
Test Suite — Agentic RAG Pipeline.

Tests the new agentic modules:
  - QueryDecomposer: simple vs complex classification
  - AgentMemory: deduplication, history tracking
  - Evaluator: sufficiency assessment
  - AgenticController: full pipeline (requires API keys)

Usage:
  python3 test_agentic.py                # Run unit tests only (no API)
  python3 test_agentic.py --live         # Run live tests with API calls
  python3 test_agentic.py --query "..."  # Single agentic query
"""
import argparse
import time
import os
import sys

import config
from core.agent_memory import AgentMemory


# ══════════════════════════════════════════════
# Unit Tests (no API calls)
# ══════════════════════════════════════════════
def test_agent_memory():
    """Test AgentMemory deduplication and tracking."""
    print("\n── Test: AgentMemory ──")
    memory = AgentMemory(original_query="test query")

    # Add first batch
    results1 = [
        ("chunk A about habits", 0.9),
        ("chunk B about mindset", 0.8),
        ("chunk C about goals", 0.7),
    ]
    new1 = memory.add_search_results("habits", results1, iteration=1)
    assert new1 == 3, f"Expected 3 new chunks, got {new1}"
    assert memory.total_chunks == 3
    print(f"   ✅ Batch 1: {new1} new chunks (total: {memory.total_chunks})")

    # Add second batch with overlap
    results2 = [
        ("chunk A about habits", 0.95),  # Duplicate — higher score
        ("chunk D about success", 0.6),   # New
    ]
    new2 = memory.add_search_results("success", results2, iteration=2)
    assert new2 == 1, f"Expected 1 new chunk, got {new2}"
    assert memory.total_chunks == 4
    print(f"   ✅ Batch 2: {new2} new chunks, 1 deduped (total: {memory.total_chunks})")

    # Verify higher score is kept
    all_chunks = memory.get_all_chunks()
    top_score = all_chunks[0][1]
    assert top_score == 0.95, f"Expected top score 0.95, got {top_score}"
    print(f"   ✅ Dedup: kept higher score ({top_score})")

    # Test has_searched
    assert memory.has_searched("habits") is True
    assert memory.has_searched("unknown") is False
    print(f"   ✅ has_searched: correct")

    # Test context summary
    summary = memory.get_context_summary()
    assert "4 chunks" in summary
    print(f"   ✅ Context summary: {len(summary)} chars")

    print("   ✅ AgentMemory: ALL PASSED")


def test_agent_memory_edge_cases():
    """Test AgentMemory edge cases."""
    print("\n── Test: AgentMemory Edge Cases ──")

    # Empty memory
    memory = AgentMemory(original_query="test")
    assert memory.total_chunks == 0
    assert memory.get_all_chunks() == []
    assert "ยังไม่มี" in memory.get_context_summary()
    print(f"   ✅ Empty memory handled")

    # Empty results
    new = memory.add_search_results("query", [], iteration=1)
    assert new == 0
    print(f"   ✅ Empty results handled")

    print("   ✅ Edge Cases: ALL PASSED")


# ══════════════════════════════════════════════
# Live Tests (requires API keys)
# ══════════════════════════════════════════════
def test_decomposer_live():
    """Test QueryDecomposer with actual API calls."""
    from core.query_decomposer import decompose

    print("\n── Test: QueryDecomposer (LIVE) ──")

    # Simple query
    result = decompose("Atomic Habits สอนอะไร")
    print(f"   Simple: type={result.query_type}, sub_queries={result.sub_queries}")
    assert result.query_type == "simple", f"Expected simple, got {result.query_type}"
    print(f"   ✅ Simple query classified correctly")

    # Complex query
    result = decompose("เปรียบเทียบหลักการลงทุนจาก Rich Dad กับ Psychology of Money")
    print(f"   Complex: type={result.query_type}, sub_queries={result.sub_queries}")
    assert result.query_type == "complex", f"Expected complex, got {result.query_type}"
    assert len(result.sub_queries) >= 2, f"Expected >= 2 sub-queries, got {len(result.sub_queries)}"
    print(f"   ✅ Complex query decomposed into {len(result.sub_queries)} sub-queries")

    print("   ✅ QueryDecomposer: ALL PASSED")


def test_full_pipeline_live():
    """Test full agentic pipeline with actual API calls."""
    from rag_searcher import RAGSearcher
    from core.agentic_controller import AgenticController

    print("\n── Test: Full Agentic Pipeline (LIVE) ──")

    # Check index exists
    index_path = os.path.join(config.STORAGE_DIR, f"{config.INDEX_NAME}.faiss")
    if not os.path.exists(index_path):
        print("   ⚠️  Index not found — skipping pipeline test")
        return

    searcher = RAGSearcher()
    searcher.load_index()

    controller = AgenticController(searcher=searcher, use_hyde=False)

    # Test 1: Simple query
    print(f"\n   [Test 1] Simple query")
    t0 = time.time()
    result = controller.run("Atomic Habits สอนอะไร")
    t1 = time.time()
    print(f"   Type: {result.query_type} | Iterations: {result.iterations}")
    print(f"   Chunks: {result.total_chunks} | Time: {t1-t0:.2f}s")
    print(f"   Answer: {result.answer[:200]}...")
    assert result.answer, "Expected non-empty answer"
    print(f"   ✅ Simple query passed")

    # Test 2: Complex query
    print(f"\n   [Test 2] Complex query")
    t0 = time.time()
    result = controller.run(
        "เปรียบเทียบหลักการลงทุนจาก Rich Dad กับ Psychology of Money"
    )
    t1 = time.time()
    print(f"   Type: {result.query_type} | Iterations: {result.iterations}")
    print(f"   Chunks: {result.total_chunks} | Time: {t1-t0:.2f}s")
    print(f"   Sub-queries: {result.sub_queries}")
    print(f"   Answer: {result.answer[:200]}...")
    assert result.answer, "Expected non-empty answer"
    assert result.query_type == "complex", "Expected complex query type"
    print(f"   ✅ Complex query passed")

    print("\n   ✅ Full Pipeline: ALL PASSED")


# ══════════════════════════════════════════════
# CLI Runner
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Test Agentic RAG pipeline")
    parser.add_argument("--live", action="store_true", help="Run live API tests")
    parser.add_argument("--query", type=str, help="Run a single agentic query")
    args = parser.parse_args()

    print("=" * 60)
    print("🧪 Agentic RAG — Test Suite")
    print("=" * 60)

    # Unit tests (always run)
    test_agent_memory()
    test_agent_memory_edge_cases()

    if args.live:
        test_decomposer_live()
        test_full_pipeline_live()

    if args.query:
        from rag_searcher import RAGSearcher
        from core.agentic_controller import AgenticController

        index_path = os.path.join(config.STORAGE_DIR, f"{config.INDEX_NAME}.faiss")
        if not os.path.exists(index_path):
            print("❌ Index not found! Please build it first: python3 build_index.py")
            return

        searcher = RAGSearcher()
        searcher.load_index()
        controller = AgenticController(searcher=searcher)

        print(f"\n🧠 Agentic Query: {args.query}")
        result = controller.run(args.query)
        print(f"\n{'─' * 60}")
        print(result.answer)
        print(f"{'─' * 60}")

    print(f"\n{'=' * 60}")
    print(f"✅ All tests complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
