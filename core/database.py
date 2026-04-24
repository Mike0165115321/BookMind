import sqlite3
import os
from datetime import datetime
import config

class DatabaseManager:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(config.STORAGE_DIR, "metadata.db")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize the database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Table for documents
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    book_title TEXT,
                    category TEXT,
                    upload_date TEXT,
                    status TEXT DEFAULT 'WAITING',
                    total_chunks INTEGER DEFAULT 0,
                    error_message TEXT
                )
            ''')
            conn.commit()

    def add_document(self, filename, book_title=None, category=None):
        """Add a new document entry to the database."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO documents (filename, book_title, category, upload_date, status)
                VALUES (?, ?, ?, ?, 'WAITING')
            ''', (filename, book_title, category, now))
            conn.commit()
            return cursor.lastrowid

    def update_status(self, doc_id, status, total_chunks=0, error_message=None):
        """Update the ingestion status of a document."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE documents 
                SET status = ?, total_chunks = ?, error_message = ?
                WHERE id = ?
            ''', (status, total_chunks, error_message, doc_id))
            conn.commit()

    def get_all_documents(self):
        """Get list of all documents and their status."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM documents ORDER BY upload_date DESC')
            return [dict(row) for row in cursor.fetchall()]

    def get_document(self, doc_id):
        """Get a single document entry by ID."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_document(self, doc_id):
        """Remove a document entry."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
            conn.commit()

# Singleton instance
db = DatabaseManager()
