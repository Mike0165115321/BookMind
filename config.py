"""
Central configuration for the RAG system.
Edit this file to change paths, models, or tuning parameters.
All other modules import from here — single source of truth.
"""
import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

# ──────────────────────────────────────────────
# Models (local paths — no internet required)
# ──────────────────────────────────────────────
MODEL_EMBEDDING = "/home/mikedev/MyModels/Model-RAG/intfloat-multilingual-e5-large"
MODEL_RERANKER = "/home/mikedev/MyModels/Model-RAG/BAAI-bge-reranker-v2-m3"

# ──────────────────────────────────────────────
# Index
# ──────────────────────────────────────────────
INDEX_NAME = "RAG_system"

# ──────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────
CHUNK_SIZE = 500       # Max characters per chunk
CHUNK_OVERLAP = 100    # Overlap between consecutive chunks (prevents info loss at boundaries)

# ──────────────────────────────────────────────
# Hybrid Search weights (must sum to 1.0)
# ──────────────────────────────────────────────
HYBRID_DENSE_WEIGHT = 0.7    # Weight for Dense/FAISS (semantic meaning)
HYBRID_BM25_WEIGHT = 0.3     # Weight for BM25 (keyword matching)

# ──────────────────────────────────────────────
# Search tuning
# ──────────────────────────────────────────────
TOP_K_RETRIEVAL = 10   # Number of candidates from FAISS (Bi-Encoder stage)
TOP_K_DISPLAY = 5      # Number of final results shown to user (after Reranking)
BATCH_SIZE = 32        # Embedding batch size (adjust based on VRAM)
