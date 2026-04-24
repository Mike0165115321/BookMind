"""
Entry point: Interactive search mode.

Usage:
    python3 search.py
"""
import os
import config
from rag_searcher import RAGSearcher


def main():
    # Verify index exists
    index_path = os.path.join(config.STORAGE_DIR, f"{config.INDEX_NAME}.faiss")
    if not os.path.exists(index_path):
        print("❌ Index not found! Please build it first:")
        print("   python3 build_index.py")
        return

    # Load searcher
    searcher = RAGSearcher()
    searcher.load_index(storage_dir=config.STORAGE_DIR, index_name=config.INDEX_NAME)

    # Interactive loop
    print(f"\n{'=' * 50}")
    print(f"🤖 RAG Search — {config.INDEX_NAME}")
    print(f"📊 Showing top {config.TOP_K_DISPLAY} results (from {config.TOP_K_RETRIEVAL} candidates)")
    print(f"{'=' * 50}")

    while True:
        try:
            query = input("\n🔎 ป้อนคำถาม (หรือพิมพ์ 'exit'): ")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 ลาก่อน!")
            break

        if query.strip().lower() in ("exit", "quit", "q"):
            print("👋 ลาก่อน!")
            break

        if not query.strip():
            continue

        results = searcher.search(query, top_k=config.TOP_K_RETRIEVAL)

        print(f"\n🎯 ผลลัพธ์ที่เกี่ยวข้องที่สุด:")
        if not results:
            print("   ไม่พบข้อมูลที่เกี่ยวข้อง")
            continue

        for i, (doc, score) in enumerate(results[:config.TOP_K_DISPLAY]):
            print(f"[{i+1}] (Score: {score:.4f}) -> {doc[:200]}...")


if __name__ == "__main__":
    main()
