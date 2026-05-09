
import os
import sys
import pickle
import sqlite3

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import db
import config

def fix_chunk_counts():
    print("🔍 Reading RAG_system_data.pkl...")
    data_path = os.path.join(config.STORAGE_DIR, f"{config.INDEX_NAME}_data.pkl")
    
    if not os.path.exists(data_path):
        print(f"❌ Index data not found at {data_path}")
        return

    with open(data_path, "rb") as f:
        chunks = pickle.load(f)
    
    print(f"📂 Total chunks in index: {len(chunks)}")
    
    # Get all docs from DB
    docs = db.get_all_documents()
    
    for doc in docs:
        filename = doc['filename']
        book_title = doc['book_title'] or ""
        
        # Count chunks that belong to this file
        # Chunks usually start with [Title] or similar prefix
        count = 0
        search_term = book_title if book_title else filename
        
        for c in chunks:
            if search_term in c:
                count += 1
        
        if count > 0:
            db.update_status(doc['id'], "completed", total_chunks=count)
            print(f"✅ Updated {filename}: {count} chunks")
        else:
            print(f"⚠️ No chunks found for {filename} in index data.")

if __name__ == "__main__":
    fix_chunk_counts()
