"""
Entry point: Build or rebuild the FAISS index from source data.

Usage:
    python3 build_index.py          # Build only if index doesn't exist
    python3 build_index.py --force  # Force rebuild (delete old index first)
"""
import argparse
import os
import shutil
import config
from rag_creator import RAGCreator


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from source data")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if index exists")
    args = parser.parse_args()

    index_path = os.path.join(config.STORAGE_DIR, f"{config.INDEX_NAME}.faiss")

    # Check if rebuild is needed
    if os.path.exists(index_path) and not args.force:
        print(f"✅ Index already exists at: {index_path}")
        print("💡 Use --force to rebuild: python3 build_index.py --force")
        return

    # Clean old storage if forcing
    if args.force and os.path.exists(config.STORAGE_DIR):
        print("🗑️  Removing old index...")
        shutil.rmtree(config.STORAGE_DIR)

    # Build
    print("=" * 50)
    print("🔨 RAG Index Builder")
    print("=" * 50)
    creator = RAGCreator()
    success = creator.build_and_save(
        source_path=config.DATA_DIR,
        save_dir=config.STORAGE_DIR,
        index_name=config.INDEX_NAME
    )

    if success:
        print("\n🎉 Index built successfully!")
    else:
        print("\n❌ Failed to build index.")


if __name__ == "__main__":
    main()
