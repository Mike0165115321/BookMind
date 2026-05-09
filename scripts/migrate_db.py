
import os
import sqlite3
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config

def migrate():
    db_path = os.path.join(config.DATA_DIR, "bookmind.db")
    print(f"🔧 Migrating database at {db_path}...")
    
    if not os.path.exists(db_path):
        print("❌ Database file not found!")
        return

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("ALTER TABLE documents ADD COLUMN total_chunks INTEGER DEFAULT 0;")
        conn.commit()
        print("✅ Column total_chunks added successfully!")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("ℹ️ Column already exists, skipping.")
        else:
            print(f"❌ Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
