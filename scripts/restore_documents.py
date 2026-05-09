
import os
import sys
import sqlite3

# Add parent directory to path so we can import core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import db
import config

def restore():
    print("🔍 Scanning uploads folder...")
    upload_dir = os.path.join(config.DATA_DIR, "uploads")
    if not os.path.exists(upload_dir):
        print("❌ Uploads directory not found.")
        return

    files = [f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]
    print(f"📂 Found {len(files)} files.")

    # Get existing filenames to avoid duplicates
    existing_docs = db.get_all_documents()
    existing_filenames = {d['filename'] for d in existing_docs}

    count = 0
    for filename in files:
        if filename not in existing_filenames:
            # Add to DB as 'completed' assuming they were already processed before
            db.add_document(filename=filename, book_title=filename.replace(".pdf", "").replace(".md", ""), category="Restored")
            # Get the ID of the last added doc to update its status
            cursor = db.get_connection().execute("SELECT id FROM documents WHERE filename = ?", (filename,))
            row = cursor.fetchone()
            if row:
                db.update_document_status(row['id'], "completed")
            print(f"✅ Restored: {filename}")
            count += 1
        else:
            print(f"⏩ Skipped (already exists): {filename}")

    print(f"🎉 Done! Restored {count} documents to the database.")

if __name__ == "__main__":
    restore()
