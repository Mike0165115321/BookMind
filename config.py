
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
MODEL_RERANKER = "/home/mikedev/MyModels/Model-RAG/BAAI-bge-reranker-base"

# ──────────────────────────────────────────────
# Index
# ──────────────────────────────────────────────
INDEX_NAME = "RAG_system"

# ──────────────────────────────────────────────
# Search tuning
# ──────────────────────────────────────────────
TOP_K_RETRIEVAL = 10   # Number of candidates from FAISS (Bi-Encoder stage)
TOP_K_DISPLAY = 5     # Number of final results shown to user (after Reranking)
BATCH_SIZE = 32        # Embedding batch size (adjust based on VRAM)
