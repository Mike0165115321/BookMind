"""
Central configuration for the RAG system.
Edit this file to change paths, models, or tuning parameters.
All other modules import from here — single source of truth.
"""
import os

# ──────────────────────────────────────────────
# Application Server
# ──────────────────────────────────────────────
APP_HOST = "0.0.0.0"
APP_PORT = 8081        # Change this if 8000 is occupied

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

# ──────────────────────────────────────────────
# Models (local paths — no internet required)
# ──────────────────────────────────────────────
# Try to use local paths if they exist (for speed), otherwise fallback to Hugging Face model names
_LOCAL_EMBEDDING = "/home/mikedev/MyModels/Model-RAG/intfloat-multilingual-e5-large"
_LOCAL_RERANKER = "/home/mikedev/MyModels/Model-RAG/BAAI-bge-reranker-v2-m3"

MODEL_EMBEDDING = _LOCAL_EMBEDDING if os.path.exists(_LOCAL_EMBEDDING) else "intfloat/multilingual-e5-large"
MODEL_RERANKER = _LOCAL_RERANKER if os.path.exists(_LOCAL_RERANKER) else "BAAI/bge-reranker-v2-m3"

# ──────────────────────────────────────────────
# Index
# ──────────────────────────────────────────────
INDEX_NAME = "RAG_system"

# ──────────────────────────────────────────────
# Dynamic Settings & Tuning (Fetched dynamically from SQLite db)
# ──────────────────────────────────────────────
_DEFAULTS = {
    "CHUNK_SIZE": 500,
    "CHUNK_OVERLAP": 100,
    "HYBRID_DENSE_WEIGHT": 0.7,
    "HYBRID_BM25_WEIGHT": 0.3,
    "RERANK_SCORE_GAP": 0.15,
    "TOP_K_RETRIEVAL": 10,
    "TOP_K_DISPLAY": 5,
    "RELEVANCE_THRESHOLD": 0.15,
    "BATCH_SIZE": 32,
    "COMPRESSION_ENABLED": False,
    "COMPRESSION_EMBEDDING_THRESHOLD": 0.45,
    "COMPRESSION_TOP_N_SIMPLE": 5,
    "COMPRESSION_TOP_N_COMPLEX": 12,
    "COMPRESSION_MIN_SENTENCE_LENGTH": 10,
    "AGENTIC_MAX_ITERATIONS": 3,
    "AGENTIC_SUFFICIENCY_THRESHOLD": 0.7,
    "AGENTIC_MAX_CHUNKS": 20,
}

import sqlite3

def _get_setting_from_db(key, default):
    db_path = os.path.join(DATA_DIR, "bookmind.db")
    if not os.path.exists(db_path):
        return default
    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row and row[0] is not None:
                val = str(row[0])
                if val.lower() == 'true': return True
                if val.lower() == 'false': return False
                try:
                    if '.' in val:
                        return float(val)
                    return int(val)
                except ValueError:
                    return val
            return default
    except Exception:
        return default

def __getattr__(name):
    if name in _DEFAULTS:
        return _get_setting_from_db(name.lower(), _DEFAULTS[name])
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# ──────────────────────────────────────────────
# LLM Generation (Gemini) & Query Transform (Groq)
# ──────────────────────────────────────────────
# Model settings are in core/config.py (loaded from .env)
# API keys are managed by core/key_manager.py (round-robin rotation)
from core.config import settings
GEMINI_MODEL = settings.GEMINI_MODEL       # gemini-2.5-flash
GROQ_MODEL = settings.GROQ_MODEL           # llama-3.3-70b-versatile
ENABLE_HYDE = True                          # Enable HyDE query transform


