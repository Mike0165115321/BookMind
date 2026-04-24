# 📚 RAG System — Full Pipeline Knowledge Base

ระบบค้นหาและตอบคำถามจากหนังสืออัจฉริยะ ผสม **HyDE Query Transform** + **Hybrid Search** + **Adaptive Reranking** + **Gemini LLM Generation** + **🧠 Agentic RAG** (Query Decomposition & Multi-hop Retrieval) พร้อม **Web UI** แบบ real-time streaming

> **สถานะ:** ✅ Full RAG Pipeline + Agentic RAG สมบูรณ์ (v3.0)  
> **Version:** 3.0 — Agentic RAG + HyDE + Hybrid Search + Adaptive Reranking + Gemini Generation + Web UI

📘 **[เอกสารเทคนิคฉบับเต็ม (Technical Guide)](docs/technical_guide.md)**

---

## 📸 Screenshots

### หน้า Welcome — Dark Theme + Suggestion Chips
![Web UI Welcome](docs/images/web-ui-welcome.png)

### ผลลัพธ์การค้นหา — คำตอบ + แหล่งอ้างอิง (ไม่เปิด HyDE)
![Web UI Answer](docs/images/web-ui-answer.png)

### ผลลัพธ์ด้วย HyDE — Timing ครบทุก Stage
![Web UI Timing](docs/images/web-ui-timing.png)

---

## 🏗️ Architecture

ระบบรองรับ 2 โหมด:
- **Classic:** HyDE → Search → Generate (single-shot)
- **🧠 Agentic:** Decompose → Multi-hop Search → Evaluate → Generate

### Classic Pipeline

```
User Query
    │
    ▼
┌──────────────────────────┐
│  Stage 0: HyDE Transform │
│  (Groq LLaMA 3.3 70B)   │
│  สร้างคำตอบสมมติเพื่อ     │
│  ปรับปรุงความแม่นยำค้นหา   │
└────────────┬─────────────┘
             ▼
    ┌────────┴────────┐
    ▼                 ▼
┌────────────┐  ┌──────────┐
│Dense Search│  │BM25 Search│
│ (FAISS+GPU)│  │  (CPU)   │
│ e5-large   │  │ rank-bm25│
└─────┬──────┘  └─────┬────┘
      │ 70%           │ 30%
      └───────┬───────┘
              ▼
     ┌────────────────┐
     │ Adaptive Gate  │
     │ gap > 0.15 → ⚡│
     │ gap ≤ 0.15 → 🔬│
     └───┬───────┬────┘
    ⚡Skip   🔬Rerank
         └───┬───┘
             ▼
   Gemini LLM Generation
   SSE Streaming → Web UI
```

### 🧠 Agentic Pipeline (Multi-hop)

```
User Query (complex, multi-book)
    │
    ▼
┌──────────────────────────┐
│  Decompose (Groq LLM)   │ "เปรียบเทียบ A กับ B"
│  → sub-query 1: "A"     │  → แยกเป็น 2 คำถามย่อย
│  → sub-query 2: "B"     │
└────────────┬─────────────┘
             ▼
    ┌────────┴─────────┐
    ▼                  ▼
 [HyDE → Search 1]  [HyDE → Search 2]
    │                  │
    └────────┬─────────┘
             ▼
┌──────────────────────────┐
│  Evaluate (Groq LLM)    │ ข้อมูลครบหรือยัง?
│  confidence ≥ 0.7 → ✅   │ → ครบ! สร้างคำตอบ
│  confidence < 0.7 → 🔄   │ → ยังไม่ครบ ค้นเพิ่ม
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│  Balanced Chunk Select   │ round-robin จากทุก source
│  → Gemini Generation     │ สังเคราะห์ข้ามเล่ม
└──────────────────────────┘
```

### Pipeline Stages

| Stage | Method | หน้าที่ | ทำงานบน |
|-------|--------|---------|---------|
| **0** | HyDE (Groq LLaMA) | สร้างเอกสารสมมติเพื่อปรับปรุงคำค้น | Cloud API |
| **1a** | Dense (FAISS) | จับ "ความหมาย" — คำต่างกันแต่หมายถึงเรื่องเดียวกัน | GPU |
| **1b** | BM25 (Sparse) | จับ "คำตรงกัน" — ชื่อคน, ชื่อหนังสือ, ศัพท์เฉพาะ | CPU |
| **2** | Score Merge | รวม Dense (70%) + BM25 (30%) แล้ว normalize | CPU |
| **3** | Adaptive Reranker | ⚡ Skip ถ้าชัด / 🔬 Rerank ถ้ากำกวม (gap ≤ 0.15) | GPU |
| **4** | Gemini Generation | สร้างคำตอบ SSE streaming จากเนื้อหาที่ค้นเจอ | Cloud API |
| **A1** | Query Decomposer | 🧠 แตกคำถามซับซ้อนเป็น sub-queries (Agentic) | Cloud API |
| **A2** | Evaluator | 📊 ประเมินว่าข้อมูลครบหรือยัง + สร้าง follow-up (Agentic) | Cloud API |
| **A3** | Balanced Selection | ⚖️ round-robin เลือก chunks จากทุก source (Agentic) | CPU |

---

## 🧠 Models

| Role | Model | ขนาด | ทำงานบน |
|------|-------|-------|---------|
| **Embedding** | `intfloat/multilingual-e5-large` | ~2.2 GB | GPU (local) |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | ~2.2 GB | GPU (local) |
| **LLM Generation** | `Gemini 2.5 Flash` | — | Cloud API |
| **HyDE Transform** | `Groq LLaMA 3.3 70B` | — | Cloud API |

- โมเดล Embedding + Reranker เก็บในเครื่อง (`~/MyModels/Model-RAG/`)
- LLM ใช้ API keys แบบ round-robin (10 Gemini + 3 Groq keys)

---

## ✂️ Chunking Strategy

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
├── config.py               # ⚙️  Central config (paths, models, tuning, agentic)
├── rag_creator.py          # 🔨 Chunking + embedding + index building
├── rag_searcher.py         # 🔍 Hybrid search + adaptive reranking
├── build_index.py          # ▶️  CLI: build/rebuild index
├── search.py               # ▶️  CLI: interactive search (retrieval only)
├── ask.py                  # 🤖 CLI: full RAG pipeline (Classic + Agentic)
├── web_server.py           # 🌐 FastAPI + SSE streaming (Classic + Agentic)
├── test_rag.py             # ✅ Test suite (search)
├── test_agentic.py         # 🧪 Test suite (agentic pipeline)
│
├── core/                   # 📦 Core modules
│   ├── __init__.py
│   ├── config.py           #   🔐 .env loader (API keys, model settings)
│   ├── key_manager.py      #   🔑 Round-robin API key rotation
│   ├── llm_generator.py    #   🤖 Gemini LLM generation (sync + streaming)
│   ├── query_transformer.py#   🪄 HyDE + Query Rewriting (via Groq)
│   ├── query_decomposer.py #   🧠 Query Decomposition (simple/complex → sub-queries)
│   ├── evaluator.py        #   📊 Sufficiency Evaluator (confidence + follow-up)
│   ├── agent_memory.py     #   💾 Working Memory (dedup + balanced selection)
│   └── agentic_controller.py#  🔄 Agentic Orchestrator (decompose → search → eval → loop)
│
├── web/                    # 🎨 Frontend (Dark theme chat UI)
│   ├── index.html          #   📄 Main page (HyDE + Agentic toggles)
│   ├── style.css           #   🎨 Dark theme + agentic steps UI
│   └── app.js              #   ⚡ SSE streaming + agentic event handling
│
├── data/                   # 📂 Source .jsonl files (120 files, 3,002+ entries)
├── storage/                # 💾 FAISS + BM25 + text data indices
│   ├── RAG_system.faiss
│   ├── RAG_system_data.pkl
│   └── RAG_system_bm25.pkl
├── .env                    # 🔐 API keys (Gemini x10, Groq x3)
└── venv/                   # Python virtual environment
```

---

## 🚀 Quick Start

### 1. Build Index
```bash
python3 build_index.py            # สร้าง index ครั้งแรก
python3 build_index.py --force    # สร้างใหม่ (ลบของเก่า)
```

### 2. CLI — Classic Pipeline
```bash
python3 ask.py                         # Interactive mode
python3 ask.py "สามก๊กสอนอะไร"          # Single question
python3 ask.py --no-hyde "วิธีสร้างนิสัย" # ไม่ใช้ HyDE
python3 ask.py --no-stream "Growth Mindset" # ไม่ streaming
```

### 3. CLI — 🧠 Agentic Pipeline (Multi-hop)
```bash
python3 ask.py --agentic "เปรียบเทียบ Rich Dad กับ Psychology of Money"
python3 ask.py --agentic "วิเคราะห์ความเชื่อมโยงระหว่าง Atomic Habits กับ 7 Habits"
python3 ask.py --agentic --no-hyde ".."  # Agentic ไม่ใช้ HyDE
```

### 4. Web UI
```bash
python3 web_server.py
# → Open http://localhost:8000
# → เปิด toggle 🧠 Agentic สำหรับคำถามซับซ้อน
```

### 5. Search Only (ไม่ต่อ LLM)
```bash
python3 search.py
python3 test_rag.py
python3 test_agentic.py             # Unit tests (agentic)
python3 test_agentic.py --live      # Live API tests
```

---

## ⚙️ Configuration

### RAG Tuning — `config.py`

```python
# Hybrid Search weights (must sum to 1.0)
HYBRID_DENSE_WEIGHT = 0.7    # Semantic meaning
HYBRID_BM25_WEIGHT  = 0.3    # Keyword matching

# Adaptive Reranking
RERANK_SCORE_GAP = 0.15      # Skip reranker if gap > threshold

# Search tuning
TOP_K_RETRIEVAL = 10    # FAISS candidates
TOP_K_DISPLAY   = 5     # Final results shown
ENABLE_HYDE     = True  # HyDE query transform on/off

# 🧠 Agentic RAG
AGENTIC_MAX_ITERATIONS = 3        # Max search loop iterations
AGENTIC_SUFFICIENCY_THRESHOLD = 0.7  # Stop searching when confidence ≥ 0.7
AGENTIC_MAX_CHUNKS = 20           # Max total chunks across all iterations
```

### LLM & API Keys — `core/config.py` + `.env`

```bash
# .env
GEMINI_API_KEYS='key1,key2,...'    # Round-robin rotation
GROQ_API_KEYS='key1,key2,...'

# Optional overrides
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.3
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.7
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

| Metric | Classic | 🧠 Agentic (complex) |
|--------|---------|---------------------|
| Total Chunks | 5,738 | 5,738 |
| Embedding Dimension | 1,024 | 1,024 |
| Search Latency (skip rerank) | ~15ms | ~15ms × N sub-queries |
| Search Latency (with rerank) | ~300ms | ~300ms × N sub-queries |
| HyDE Transform | ~1.5s | ~1.5s × N sub-queries |
| LLM Generation | ~5-8s | ~8-15s (more context) |
| API Calls | 2 (HyDE + Gen) | 4-8 (Decompose + HyDE×N + Eval + Gen) |
| **Total** | **~7-10s** | **~15-30s** |

> 💡 คำถาม simple ใน Agentic mode จะ bypass ไป Classic pipeline → ไม่มี overhead เพิ่ม

---

## 🗺️ Roadmap

- [x] Dense Search (FAISS + e5-large)
- [x] Cross-Encoder Reranking (bge-reranker-v2-m3)
- [x] Intelligent Chunking (500 chars + 100 overlap)
- [x] Hybrid Search (Dense + BM25)
- [x] Adaptive Reranking (score-gap based skip/rerank)
- [x] LLM Generation (Gemini 2.5 Flash)
- [x] Query Transform (HyDE via Groq LLaMA 3.3 70B)
- [x] Web UI (FastAPI + SSE + Dark Theme)
- [x] API Key Rotation (round-robin)
- [x] 🧠 Agentic RAG — Query Decomposition + Multi-hop Retrieval
- [x] 📊 Sufficiency Evaluator — ประเมินข้อมูลครบหรือยัง
- [x] ⚖️ Balanced Chunk Selection — round-robin จากทุก source
- [ ] Conversation Memory (multi-turn)
- [ ] Document Upload (PDF/TXT via Web UI)
- [ ] Multi-Agent System (specialized agents per domain)

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.12 |
| Embedding | sentence-transformers + e5-large |
| Vector DB | FAISS (GPU-accelerated) |
| Sparse Search | rank-bm25 |
| Reranker | CrossEncoder (bge-reranker-v2-m3) |
| LLM Generation | Gemini 2.5 Flash (via google-genai) |
| Query Transform | Groq LLaMA 3.3 70B |
| 🧠 Agentic RAG | Query Decomposition + Multi-hop + Evaluator |
| API Key Management | Round-robin rotation (KeyManager) |
| Web Backend | FastAPI + uvicorn |
| Web Frontend | Vanilla HTML/CSS/JS + SSE |
| Streaming | Server-Sent Events (SSE) |
| GPU | NVIDIA RTX 4060 (CUDA) |
| Data Format | JSONL |
