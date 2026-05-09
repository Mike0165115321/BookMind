
import os
import sys
import asyncio
import sqlite3
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import db
from core.ingestor import Ingestor
import config

async def rebuild():
    print("🧹 [1/3] Clearing old data...")
    # 1. Delete storage files
    if os.path.exists(config.STORAGE_DIR):
        for f in os.listdir(config.STORAGE_DIR):
            if f.startswith(config.INDEX_NAME):
                os.remove(os.path.join(config.STORAGE_DIR, f))
                print(f"   🗑️ Removed index file: {f}")

    # 2. Clear documents table
    with db.get_connection() as conn:
        conn.execute("DELETE FROM documents")
        conn.commit()
    print("   ✅ Documents table cleared.")

    # 3. Scan and re-ingest
    print("🏗️ [2/3] Scanning and Re-ingesting files...")
    upload_dir = os.path.join(config.DATA_DIR, "uploads")
    if not os.path.exists(upload_dir):
        print("❌ Uploads directory not found.")
        return

    files = [f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]
    print(f"📂 Found {len(files)} files to process.")

    ingestor = Ingestor()
    
    for filename in files:
        filepath = os.path.join(upload_dir, filename)
        # Add to DB
        doc_id = db.add_document(
            filename=filename, 
            book_title=filename.replace(".pdf", "").replace(".md", ""), 
            category="Restored"
        )
        print(f"🚀 Processing {filename} (ID: {doc_id})...")
        
        # We run it synchronously in this script for simplicity
        try:
            await ingestor.process_document(doc_id, filepath, book_title=filename.replace(".pdf", "").replace(".md", ""), category="Restored")
            print(f"   ✅ Finished {filename}")
        except Exception as e:
            print(f"   ❌ Failed {filename}: {e}")

    print("🎉 [3/3] Everything is rebuilt and synchronized!")

if __name__ == "__main__":
    asyncio.run(rebuild())
