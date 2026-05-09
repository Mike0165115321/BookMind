import sqlite3
import os
import json
from datetime import datetime
import config

DB_PATH = os.path.join(config.DATA_DIR, "bookmind.db")

class Database:
    def __init__(self):
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        with self.get_connection() as conn:
            # Table for Chat Sessions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Table for Messages
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT,
                    role TEXT,
                    content TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
                )
            """)
            # Table for Settings
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            # Table for Documents (Ingestion Tracking)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    book_title TEXT,
                    category TEXT,
                    status TEXT DEFAULT 'pending',
                    total_chunks INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Default Settings
            defaults = {
                "agentic_provider": "groq",
                "agentic_model": "llama-3.3-70b-versatile"
            }
            for key, value in defaults.items():
                conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, value))
                
            conn.commit()

    def create_chat(self, chat_id, title="New Chat"):
        with self.get_connection() as conn:
            conn.execute("INSERT INTO chats (id, title) VALUES (?, ?)", (chat_id, title))
            conn.commit()

    def delete_chat(self, chat_id):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            conn.commit()

    def add_message(self, chat_id, role, content, metadata=None):
        meta_json = json.dumps(metadata) if metadata else None
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO messages (chat_id, role, content, metadata) VALUES (?, ?, ?, ?)",
                (chat_id, role, content, meta_json)
            )
            # Update chat's updated_at
            conn.execute("UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (chat_id,))
            conn.commit()

    def get_chats(self):
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM chats ORDER BY updated_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_messages(self, chat_id):
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC", (chat_id,))
            return [dict(row) for row in cursor.fetchall()]

    # ── Document Management ────────────────────────────────────
    def add_document(self, filename, book_title=None, category=None):
        with self.get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO documents (filename, book_title, category) VALUES (?, ?, ?)",
                (filename, book_title, category)
            )
            conn.commit()
            return cursor.lastrowid

    def get_document(self, doc_id):
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_documents(self):
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM documents ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def update_status(self, doc_id, status, error_message=None, total_chunks=None):
        with self.get_connection() as conn:
            if total_chunks is not None:
                conn.execute(
                    "UPDATE documents SET status = ?, error_message = ?, total_chunks = ? WHERE id = ?",
                    (status, error_message, total_chunks, doc_id)
                )
            else:
                conn.execute(
                    "UPDATE documents SET status = ?, error_message = ? WHERE id = ?",
                    (status, error_message, doc_id)
                )
            conn.commit()

    def delete_document(self, doc_id):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()

    # ── Settings ───────────────────────────────────────────────
    def set_setting(self, key, value):
        with self.get_connection() as conn:
            conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
            conn.commit()

    def get_setting(self, key, default=None):
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row["value"] if row else default

    def get_all_settings(self):
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM settings")
            return {row["key"]: row["value"] for row in cursor.fetchall()}

db = Database()
