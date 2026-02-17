# 📚 RAG System — Hybrid Retrieval-Augmented Generation

ระบบค้นหาข้อมูลอัจฉริยะที่ผสม **Dense Search (AI Embedding)** กับ **Sparse Search (BM25 Keyword)** แล้วผ่าน **Cross-Encoder Reranking** เพื่อให้ได้ผลลัพธ์ที่แม่นยำที่สุด

> **สถานะ:** Retrieval Pipeline สมบูรณ์ (Dense + BM25 + Reranker)  
> **ถัดไป:** เชื่อมต่อ LLM (Gemini) สำหรับ Generation

📘 **[อ่านเอกสารเทคนิคฉบับเต็มโดยละเอียด (Technical Guide)](docs/technical_guide.md)** — อธิบายสถาปัตยกรรม Data Pipeline และการปรับจูนอย่างละเอียด

---

## 🏗️ Architecture

```
User Query
    │
    ├──────────────────┬────────────────────┐
    ▼                  ▼                    │
┌────────────┐   ┌──────────┐              │
│Dense Search│   │BM25 Search│              │
│ (FAISS+GPU)│   │  (CPU)   │              │
│ e5-large   │   │ rank-bm25│              │
└─────┬──────┘   └─────┬────┘              │
      │ 70%            │ 30%               │
      └───────┬────────┘                   │
              ▼                            │
     ┌────────────────┐                    │
     │  Score Merge   │                    │
     │  (Normalize +  │                    │
     │   Weighted)    │                    │
     └───────┬────────┘                    │
             ▼                             │
     ┌────────────────┐                    │
     │   Reranker     │◄───────────────────┘
     │ bge-v2-m3      │    (query + doc pairs)
     │ Cross-Encoder  │
     └───────┬────────┘
             ▼
       🎯 Final Results
```

### Two-Stage Search Pipeline

| Stage | Method | หน้าที่ | ทำงานบน |
|-------|--------|---------|---------|
| **1a** | Dense (FAISS) | จับ "ความหมาย" — คำต่างกันแต่หมายถึงเรื่องเดียวกัน | GPU |
| **1b** | BM25 (Sparse) | จับ "คำตรงกัน" — ชื่อคน, ชื่อหนังสือ, ศัพท์เฉพาะ | CPU |
| **2** | Score Merge | รวม Dense (70%) + BM25 (30%) แล้ว normalize | CPU |
| **3** | Reranker | Cross-Encoder ให้คะแนนคู่ (query, doc) อย่างละเอียด | GPU |

---

## 🧠 Models

| Role | Model | ขนาด | ภาษาไทย |
|------|-------|-------|---------|
| **Embedding** | `intfloat/multilingual-e5-large` | ~2.2 GB | ⭐⭐⭐⭐⭐ |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | ~2.2 GB | ⭐⭐⭐⭐⭐ |

โมเดลทั้งหมดเก็บไว้ในเครื่อง (`~/MyModels/Model-RAG/`) — ไม่ต้องใช้อินเทอร์เน็ตตอนรัน

---

## ✂️ Chunking Strategy

ข้อมูลต้นทาง (`.jsonl`) ถูกแบ่งด้วยหลักการ:

| Parameter | ค่า | เหตุผล |
|-----------|-----|--------|
| `CHUNK_SIZE` | 500 chars | พอดีกับ Embedding model (~1 ย่อหน้า) |
| `CHUNK_OVERLAP` | 100 chars | ป้องกันข้อมูลหายตรงรอยตัด |

**ลำดับการแบ่ง:**
1. แบ่งที่ `\n` (ย่อหน้า) ก่อน
2. ถ้ายังยาวเกิน → แบ่งที่ `.` `。` `!` `?` (จุดจบประโยค)
3. ทุก chunk แนบ `[ชื่อหนังสือ]` + `หัวข้อ` ไว้ด้านบนเสมอ

---

## 📁 Project Structure

```
RAG/
├── config.py           # ⚙️  Central config (paths, models, tuning)
├── rag_creator.py      # 🔨 Core: chunking + embedding + index building
├── rag_searcher.py     # 🔍 Core: hybrid search + reranking
├── build_index.py      # ▶️  CLI: build/rebuild FAISS + BM25 index
├── search.py           # ▶️  CLI: interactive search
├── test_rag.py         # ▶️  CLI: test with predefined queries
├── data/               # 📂 Source .jsonl files (120 files, 3,002+ entries)
├── storage/            # 💾 FAISS index + BM25 corpus + text data
│   ├── RAG_system.faiss
│   ├── RAG_system_data.pkl
│   └── RAG_system_bm25.pkl
└── venv/               # Python virtual environment
```

---

## 🚀 Quick Start

### 1. Build Index
```bash
# สร้าง index ครั้งแรก
python3 build_index.py

# สร้างใหม่ (ลบของเก่า)
python3 build_index.py --force
```

### 2. Interactive Search
```bash
python3 search.py
```

### 3. Test Queries
```bash
# รันชุดทดสอบทั้งหมด
python3 test_rag.py

# ทดสอบคำถามเดียว
python3 test_rag.py --query "สามก๊กสอนอะไร"

# แสดงผลมากขึ้น
python3 test_rag.py --query "วิธีสร้างนิสัย" --top_k 10
```

---

## ⚙️ Configuration

แก้ไขทุก setting ได้ที่ `config.py` — ไฟล์เดียว มีผลทุกที่:

```python
# Models
MODEL_EMBEDDING = "/home/mikedev/MyModels/Model-RAG/intfloat-multilingual-e5-large"
MODEL_RERANKER  = "/home/mikedev/MyModels/Model-RAG/BAAI-bge-reranker-v2-m3"

# Chunking
CHUNK_SIZE    = 500     # Max chars per chunk
CHUNK_OVERLAP = 100     # Overlap between chunks

# Hybrid Search weights (must sum to 1.0)
HYBRID_DENSE_WEIGHT = 0.7    # Semantic meaning
HYBRID_BM25_WEIGHT  = 0.3    # Keyword matching

# Search tuning
TOP_K_RETRIEVAL = 10    # FAISS candidates
TOP_K_DISPLAY   = 5     # Final results shown
BATCH_SIZE      = 32    # Embedding batch size
```

---

## 💾 VRAM Usage (RTX 4060 — 8 GB)

| Component | VRAM |
|-----------|------|
| e5-large (Embedding) | ~2.2 GB |
| bge-reranker-v2-m3 | ~2.2 GB |
| FAISS Index | ~0.01 GB |
| BM25 | 0 GB (CPU only) |
| CUDA overhead | ~0.8 GB |
| **Total** | **~5.2 / 8 GB** ✅ |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Total Chunks | 5,738 |
| Embedding Dimension | 1,024 |
| Avg Tokens/Chunk (BM25) | 24 |
| Index Build Time | ~2 min |
| Search Latency | ~0.3–0.5s |

---

## 🗺️ Roadmap

- [x] Dense Search (FAISS + e5-large)
- [x] Cross-Encoder Reranking (bge-reranker-v2-m3)
- [x] Intelligent Chunking (500 chars + 100 overlap)
- [x] Hybrid Search (Dense + BM25)
- [ ] LLM Generation (Gemini API)
- [ ] Query Transform (HyDE, Query Rewriting)
- [ ] Web UI

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.12 |
| Embedding | sentence-transformers + e5-large |
| Vector DB | FAISS (GPU-accelerated) |
| Sparse Search | rank-bm25 |
| Reranker | CrossEncoder (bge-reranker-v2-m3) |
| GPU | NVIDIA RTX 4060 (CUDA) |
| Data Format | JSONL |
