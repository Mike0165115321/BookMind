# 📘 เอกสารอธิบายระบบ RAG อย่างละเอียด
# RAG System — Technical Documentation (v2.0)

> เอกสารฉบับนี้อธิบายการทำงานของระบบ RAG (Retrieval-Augmented Generation) **แบบ End-to-End** ทุกขั้นตอน
> ครอบคลุมตั้งแต่ Query Transform (HyDE), Hybrid Search, Adaptive Reranking, LLM Generation (Gemini), จนถึง Web UI

---

## สารบัญ

1. [ภาพรวมระบบ](#1-ภาพรวมระบบ)
2. [Data Pipeline — การจัดการข้อมูล](#2-data-pipeline--การจัดการข้อมูล)
3. [Chunking — การแบ่งเอกสาร](#3-chunking--การแบ่งเอกสาร)
4. [Embedding — การแปลงข้อความเป็น Vector](#4-embedding--การแปลงข้อความเป็น-vector)
5. [FAISS Index — ฐานข้อมูล Vector](#5-faiss-index--ฐานข้อมูล-vector)
6. [BM25 — Keyword Search](#6-bm25--keyword-search)
7. [Hybrid Search — การค้นหาแบบผสม](#7-hybrid-search--การค้นหาแบบผสม)
8. [Reranker + Adaptive Reranking — การจัดอันดับซ้ำอัจฉริยะ](#8-reranker--adaptive-reranking--การจัดอันดับซ้ำอัจฉริยะ)
9. [Search Pipeline แบบเต็ม](#9-search-pipeline-แบบเต็ม)
10. [HyDE — Query Transform](#10-hyde--query-transform)
11. [LLM Generation — Gemini](#11-llm-generation--gemini)
12. [Core Package — API Keys & Modules](#12-core-package--api-keys--modules)
13. [Web UI — FastAPI + SSE](#13-web-ui--fastapi--sse)
14. [โครงสร้างไฟล์และหน้าที่](#14-โครงสร้างไฟล์และหน้าที่)
15. [Configuration — การตั้งค่า](#15-configuration--การตั้งค่า)
16. [ทรัพยากรที่ใช้ (VRAM/RAM)](#16-ทรัพยากรที่ใช้-vramram)
17. [ข้อมูลโมเดล AI](#17-ข้อมูลโมเดล-ai)
18. [Flow Chart — ภาพรวมทุกขั้นตอน](#18-flow-chart--ภาพรวมทุกขั้นตอน)

---

## 1. ภาพรวมระบบ

### RAG คืออะไร?

**RAG (Retrieval-Augmented Generation)** คือสถาปัตยกรรม AI ที่แบ่งเป็น 2 ขั้นตอนหลัก:

1. **Retrieval (ค้นหา):** ค้นหาข้อมูลที่เกี่ยวข้องจากฐานความรู้
2. **Generation (สร้างคำตอบ):** ส่งข้อมูลที่ค้นเจอให้ LLM สรุปเป็นคำตอบ

> ✅ **ระบบปัจจุบัน** ครบทั้ง Retrieval + Generation (Full RAG Pipeline v2.0)
> HyDE (Groq) → Hybrid Search → Adaptive Reranking → Gemini Generation → Web UI

### ทำไมต้อง RAG?

| ปัญหาของ LLM ปกติ | RAG แก้ได้อย่างไร |
|-------------------|-----------------|
| ข้อมูลเก่า (training cutoff) | ค้นจากข้อมูลล่าสุดได้ |
| หลอน (Hallucination) | มีแหล่งอ้างอิงจริง |
| ไม่รู้ข้อมูลเฉพาะทาง | ค้นจากฐานความรู้ส่วนตัว |
| ข้อมูลส่วนตัวรั่วไหล | ข้อมูลอยู่ในเครื่องเท่านั้น |

---

## 2. Data Pipeline — การจัดการข้อมูล

### ข้อมูลต้นทาง

ข้อมูลเก็บอยู่ในโฟลเดอร์ `data/` เป็นไฟล์ `.jsonl` (JSON Lines) หนึ่งบรรทัด = หนึ่งรายการ

**ตัวอย่างข้อมูล (.jsonl):**
```json
{
  "book_title": "Atomic Habits",
  "title": "กฎข้อที่ 1: ทำให้มันชัดเจน",
  "content": "การสร้างนิสัยที่ดีเริ่มต้นจากการทำให้สิ่งกระตุ้นนั้นชัดเจน..."
}
```

### ขั้นตอนการอ่านข้อมูล

```
data/
├── atomic_habits.jsonl     → อ่านทุกบรรทัด
├── three_kingdoms.jsonl    → อ่านทุกบรรทัด
├── rich_dad.jsonl          → อ่านทุกบรรทัด
└── ... (120 ไฟล์)
```

**กระบวนการ:**
1. วนอ่านทุกไฟล์ `.jsonl` ในโฟลเดอร์ `data/`
2. แต่ละบรรทัด parse เป็น JSON
3. ดึง 3 field มารวมกัน:
   - `book_title` → แท็กชื่อหนังสือ เช่น `[Atomic Habits]`
   - `title` → หัวข้อย่อย
   - `content` → เนื้อหาหลัก
4. ส่งเข้า Chunking pipeline

**ไฟล์ที่เกี่ยวข้อง:** `rag_creator.py` → method `_load_jsonl()`

---

## 3. Chunking — การแบ่งเอกสาร

### ทำไมต้อง Chunk?

Embedding model (e5-large) ทำงานได้ดีที่สุดกับข้อความขนาด **~500 ตัวอักษร** ถ้าข้อความยาวเกินไป:
- Model จะ "เฉลี่ย" ความหมายจนไม่ชัดเจน
- ค้นหาไม่ตรงจุด (เจอเอกสารที่ "พอเกี่ยว" แต่ไม่ตรง)

ถ้าข้อความสั้นเกินไป:
- ขาดบริบท ไม่เข้าใจความหมาย

### Chunking Strategy ของระบบ

```
ต้นฉบับ (เนื้อหา 1 entry ใน JSONL):
┌─────────────────────────────────────────────────┐
│ [Atomic Habits]                                 │
│ กฎข้อที่ 1: ทำให้มันชัดเจน                        │
│                                                 │
│ การสร้างนิสัยที่ดีเริ่มต้นจากการทำให้สิ่งกระตุ้น   │
│ นั้นชัดเจน วิธีที่ดีที่สุดคือ...                   │
│ (ข้อความยาว 2,000 ตัวอักษร)                       │
└─────────────────────────────────────────────────┘
                    │
                    ▼  Chunker (500 chars, overlap 100)
    ┌──────────────────────────────────┐
    │ Chunk 1 (500 chars)             │
    │ [Atomic Habits]                 │
    │ กฎข้อที่ 1: ทำให้มันชัดเจน       │
    │ การสร้างนิสัยที่ดีเริ่มต้น...     │
    ├──────┤◄── overlap 100 chars     │
    │ Chunk 2 (500 chars)             │
    │ [Atomic Habits]                 │
    │ ...ต้นจากการทำให้สิ่งกระตุ้นนั้น  │
    │ ชัดเจน วิธีที่ดีที่สุดคือ...      │
    ├──────┤◄── overlap 100 chars     │
    │ Chunk 3 (500 chars)             │
    │ [Atomic Habits]                 │
    │ ...ที่สุดคือการใช้ implementation │
    │ intention โดยระบุว่า...          │
    └──────────────────────────────────┘
```

### ลำดับการตัด

1. **ตัดที่ย่อหน้า (`\n`)** — ธรรมชาติที่สุด ไม่ตัดกลางความคิด
2. **ตัดที่จุดจบประโยค (`.` `。` `!` `?`)** — ถ้าย่อหน้าเดียวยาวเกิน
3. **ตัดตามจำนวนตัวอักษร** — ทางเลือกสุดท้าย ถ้าไม่มีจุดตัดธรรมชาติ

### Overlap คืออะไร?

**Overlap** = ส่วนที่ซ้ำกันระหว่าง chunk ต่อกัน

```
Chunk 1: [AAAAAAAAAA|BBBB]
                     ↕ overlap 100 chars
Chunk 2:            [BBBB|CCCCCCCCCC]
```

**ทำไมต้องมี Overlap?**
- ป้องกัน "ข้อมูลหาย" ตรงรอยตัด
- ถ้าคำตอบอยู่ตรงรอยต่อพอดี → ยังเจอใน chunk ใดเช่นหนึ่ง

### Metadata Prefix

ทุก chunk จะมี **ชื่อหนังสือ + หัวข้อ** แนบไว้ด้านบนเสมอ:

```
[Atomic Habits]              ← book_title
กฎข้อที่ 1: ทำให้มันชัดเจน    ← title
การสร้างนิสัยที่ดีเริ่มต้น...  ← content (chunked)
```

สิ่งนี้ช่วยให้:
- Embedding จับบริบท "หนังสือเล่มไหน" ได้
- BM25 ค้นชื่อหนังสือตรงๆ ได้

**ไฟล์ที่เกี่ยวข้อง:** `rag_creator.py` → class `TextChunker`

**ค่าตั้ง:** `config.py` → `CHUNK_SIZE = 500`, `CHUNK_OVERLAP = 100`

---

## 4. Embedding — การแปลงข้อความเป็น Vector

### หลักการ

**Embedding** คือการแปลงข้อความเป็น "เวกเตอร์ตัวเลข" ที่แทนความหมาย

```
"วิธีสร้างนิสัยที่ดี"  →  [0.0234, -0.1456, 0.3421, ..., 0.0891]
                            (1,024 มิติ)
```

ข้อความที่มีความหมายใกล้เคียงกัน → จะมี vector ที่ชี้ไปทิศทางเดียวกัน:

```
"วิธีสร้างนิสัยที่ดี"        → [0.82, 0.34, ...]  ─┐
"การพัฒนาพฤติกรรมประจำวัน"  → [0.80, 0.31, ...]  ─┤ ใกล้กัน!
                                                    │
"ราคาทองคำวันนี้"          → [-0.12, 0.95, ...] ─┘ ไกลมาก!
```

### โมเดลที่ใช้: `intfloat/multilingual-e5-large`

| คุณสมบัติ | รายละเอียด |
|-----------|-----------|
| Parameters | ~560 ล้าน |
| Dimension | 1,024 |
| ภาษา | 100+ ภาษา (รวมไทย) |
| Benchmark | MTEB Top-tier |
| ขนาด | ~2.2 GB |

### E5 Prefix Convention

โมเดล E5 ใช้ prefix พิเศษเพื่อบอกว่าเป็น "เอกสาร" หรือ "คำถาม":

```python
# ตอนสร้าง index (เอกสาร)
"passage: [Atomic Habits]\nกฎข้อที่ 1..."

# ตอนค้นหา (คำถาม)
"query: วิธีสร้างนิสัยที่ดี"
```

สิ่งนี้ช่วยให้ model เข้าใจ "บทบาท" ของข้อความ → ให้ผลลัพธ์ดีขึ้น 10-15%

### กระบวนการ Encoding

```
5,738 chunks
    │
    ▼  Batch encoding (batch_size=32)
    │
    ├── Batch 1: chunks[0:32]    → embeddings[0:32]
    ├── Batch 2: chunks[32:64]   → embeddings[32:64]
    ├── ...
    └── Batch 180: chunks[5728:5738] → embeddings[5728:5738]
    │
    ▼
Matrix: (5,738 × 1,024) float32
= ~23.5 MB ข้อมูล vector
```

**ไฟล์ที่เกี่ยวข้อง:** `rag_creator.py` → `build_and_save()` ใช้ `model.encode()`

---

## 5. FAISS Index — ฐานข้อมูล Vector

### FAISS คืออะไร?

**FAISS (Facebook AI Similarity Search)** เป็นไลบรารีจาก Meta AI สำหรับค้นหา vector ที่ใกล้เคียงกันในเวลาเร็วมาก

### ชนิด Index ที่ใช้: `IndexFlatIP`

```
IndexFlatIP = Inner Product (Cosine Similarity เมื่อ normalize แล้ว)
```

| คุณสมบัติ | รายละเอียด |
|-----------|-----------|
| Algorithm | Brute-force (ค้นทุก vector) |
| Accuracy | 100% (exact search) |
| Speed | เร็วมากด้วย GPU |
| เหมาะกับ | Dataset < 1 ล้าน vectors |

### กระบวนการค้นหา

```
Query vector (1 × 1,024)
    │
    ▼  Inner Product กับทุก vector ใน index
    │
Index vectors (5,738 × 1,024)
    │
    ▼  Sort by score
    │
Top-10 results (indices + scores)
```

**เวลาค้น:** ~1-5 มิลลิวินาที (GPU) สำหรับ 5,738 vectors

### ไฟล์ที่บันทึก

```
storage/
├── RAG_system.faiss        # FAISS index (vector data)
├── RAG_system_data.pkl     # ข้อความ chunk ต้นฉบับ (5,738 entries)
└── RAG_system_bm25.pkl     # Tokenized corpus สำหรับ BM25
```

---

## 6. BM25 — Keyword Search

### BM25 คืออะไร?

**BM25 (Best Matching 25)** เป็นอัลกอริทึมค้นหาแบบ keyword ที่ดีที่สุด (คิดค้นปี 1994 แต่ยังใช้กันทั่วโลก)

### หลักการทำงาน

BM25 ให้คะแนนเอกสารตาม 3 ปัจจัย:

```
Score = TF × IDF × Length_Normalization
```

| ปัจจัย | ความหมาย | ตัวอย่าง |
|--------|---------|---------|
| **TF** (Term Frequency) | คำนี้ปรากฏในเอกสารกี่ครั้ง | "นิสัย" ปรากฏ 5 ครั้ง → คะแนนสูง |
| **IDF** (Inverse Document Frequency) | คำนี้หายากแค่ไหนในทุกเอกสาร | "Atomic Habits" อยู่แค่ไม่กี่ chunk → คะแนนสูง |
| **Length Norm** | ปรับตามความยาวเอกสาร | เอกสารสั้นที่มีคำตรง → ได้คะแนนมากกว่าเอกสารยาว |

### ตัวอย่างจริง

```
Query: "Atomic Habits"

Chunk A: "[Atomic Habits] กฎข้อที่ 1..."
  → TF: "Atomic" ✅ "Habits" ✅
  → IDF: คำว่า "Atomic Habits" หายาก → คะแนนสูงมาก!
  → Score: 8.5

Chunk B: "[สามก๊ก] กลยุทธ์สงคราม..."
  → TF: ไม่มีคำตรงกันเลย
  → Score: 0.0
```

### Tokenization สำหรับภาษาไทย

เนื่องจากภาษาไทยไม่มีช่องว่างระหว่างคำ ระบบใช้ **regex-based tokenizer**:

```python
def tokenize_thai(text):
    # จับคำไทย + คำอังกฤษ/ตัวเลข
    tokens = re.findall(r'[\u0E00-\u0E7F]+|[a-zA-Z0-9]+', text.lower())
    # กรองคำที่สั้นเกิน (1 ตัวอักษร)
    return [t for t in tokens if len(t) > 1]
```

**ตัวอย่าง:**
```
Input:  "[Atomic Habits] กฎข้อที่ 1: ทำให้มันชัดเจน"
Tokens: ["atomic", "habits", "กฎข้อที่", "ทำให้", "มัน", "ชัดเจน"]
```

> 📝 **หมายเหตุ:** tokenizer ปัจจุบันใช้ regex ซึ่ง "ดีพอ" สำหรับ BM25
> ถ้าต้องการความแม่นยำสูงขึ้น สามารถเปลี่ยนเป็น `PyThaiNLP` ในอนาคต

**ไฟล์ที่เกี่ยวข้อง:** `rag_creator.py` → `tokenize_thai()`, `rag_searcher.py` → `_bm25_search()`

---

## 7. Hybrid Search — การค้นหาแบบผสม

### ทำไมต้อง Hybrid?

| สถานการณ์ | Dense Search | BM25 | Hybrid |
|-----------|-------------|------|--------|
| "วิธีสร้างนิสัย" | ✅ เข้าใจความหมาย | ⚠️ อาจไม่มีคำตรง | ✅ |
| "Atomic Habits" | ⚠️ อาจเจอหนังสืออื่น | ✅ จับชื่อตรง | ✅ |
| "Jensen Huang พูดอะไร" | ⚠️ | ✅ จับชื่อคน | ✅ |
| "ปรัชญาเรื่องการลงทุน" | ✅ จับความหมายนามธรรม | ❌ | ✅ |

**Hybrid = ได้ข้อดีของทั้งสองโลก**

### วิธีรวมคะแนน

```
Query: "Atomic Habits สอนอะไร"
                │
        ┌───────┴───────┐
        ▼               ▼
   Dense Search     BM25 Search
   (FAISS)          (rank-bm25)
        │               │
        ▼               ▼
   Raw Scores       Raw Scores
   [0.85, 0.72,    [8.5, 0.0,
    0.68, 0.45]     6.2, 3.1]
        │               │
        ▼               ▼
   Normalize 0-1    Normalize 0-1
   [1.0, 0.68,     [1.0, 0.0,
    0.58, 0.0]      0.73, 0.36]
        │               │
        └───────┬───────┘
                ▼
        Weighted Merge
        Dense × 0.7 + BM25 × 0.3
        │
        ▼
   [1.0×0.7 + 1.0×0.3 = 1.00,    ← Chunk A (ทั้งคู่เห็นด้วย!)
    0.68×0.7 + 0.0×0.3 = 0.48,
    0.58×0.7 + 0.73×0.3 = 0.63,  ← Chunk C (BM25 ช่วยดันขึ้น!)
    0.0×0.7 + 0.36×0.3 = 0.11]
```

### Weight ที่ใช้

| Weight | ค่า | เหตุผล |
|--------|-----|--------|
| `HYBRID_DENSE_WEIGHT` | **0.7** | ข้อมูลเป็นหนังสือ → ความหมายสำคัญกว่าคำตรง |
| `HYBRID_BM25_WEIGHT` | **0.3** | สำรองไว้จับชื่อเฉพาะ |

> 💡 **Tuning Tip:** ถ้าข้อมูลมีชื่อเฉพาะเยอะ (เช่น FAQ, glossary) → เพิ่ม BM25 เป็น 0.4-0.5

### Score Normalization

ก่อนรวม Dense + BM25 ต้อง **normalize** ก่อน เพราะ:
- Dense scores: อยู่ในช่วง 0–1 (cosine similarity)
- BM25 scores: อยู่ในช่วง 0–∞ (ไม่มีขอบเขต)

**วิธี: Min-Max Normalization**
```
normalized = (score - min) / (max - min)
```

ทำให้ทั้งสองอยู่ในช่วง 0–1 ก่อนรวมด้วย weight

**ไฟล์ที่เกี่ยวข้อง:** `rag_searcher.py` → `_hybrid_merge()`, `_normalize_scores()`

**ค่าตั้ง:** `config.py` → `HYBRID_DENSE_WEIGHT`, `HYBRID_BM25_WEIGHT`

---

## 8. Reranker + Adaptive Reranking — การจัดอันดับซ้ำอัจฉริยะ

### ทำไมต้อง Rerank?

**ปัญหาของ Stage 1 (Retrieval):**
- Dense Search ใช้ **Bi-Encoder** — encode query กับ document แยกกัน → เร็ว แต่ไม่แม่นที่สุด
- BM25 นับแค่คำ → ไม่เข้าใจบริบท

**Reranker ใช้ Cross-Encoder** — ให้คะแนนคู่ (query, document) พร้อมกัน:

```
Bi-Encoder (Stage 1):              Cross-Encoder (Stage 2):

 Query → [Encoder] → vec_q          Query ──┐
                      ↕ similarity           ├→ [Encoder] → Score
 Doc   → [Encoder] → vec_d          Doc   ──┘

 เร็ว แต่ไม่แม่น 100%              ช้ากว่า แต่แม่นมาก!
```

### ทำไม Cross-Encoder แม่นกว่า?

เพราะมัน "อ่าน" query และ document **พร้อมกัน** ทำให้เข้าใจ:
- คำไหนในเอกสาร "ตอบ" คำถามโดยตรง
- บริบทรอบข้างของคำ
- ความสัมพันธ์ระหว่างคำถามกับเนื้อหา

### โมเดลที่ใช้: `BAAI/bge-reranker-v2-m3`

| คุณสมบัติ | รายละเอียด |
|-----------|-----------|
| Architecture | XLM-RoBERTa (Cross-Encoder) |
| Parameters | ~568 ล้าน |
| ภาษา | 100+ ภาษา (รวมไทย) |
| Version | v2-m3 (ล่าสุด, multilingual) |
| ขนาด | ~2.2 GB |

### กระบวนการ Reranking

```
จาก Hybrid Merge: 10 candidates
    │
    ▼  สร้างคู่ (query, doc) ทุกคู่
    │
    ├── (query, chunk_A) → Score: 0.92
    ├── (query, chunk_C) → Score: 0.87
    ├── (query, chunk_B) → Score: 0.45
    ├── ...
    │
    ▼  Sort by score (descending)
    │
    Top-5 Final Results
```

### Adaptive Reranking — ข้าม Reranker เมื่อไม่จำเป็น

ระบบ production ระดับโลกไม่ได้ rerank ทุก query — มันตรวจสอบก่อนว่า "จำเป็นไหม?"

**หลักการ:** ถ้า Hybrid Search ให้ผลลัพธ์ที่ **Top-1 ทิ้งห่าง Top-2 มาก** (score gap สูง) แสดงว่าผลลัพธ์ชัดเจนอยู่แล้ว → ข้าม Reranker ได้

```
Hybrid Merge Scores
        │
        ▼
   Gap = Top-1 - Top-2
        │
   ┌────┴────┐
   ▼         ▼
 Gap > 0.15  Gap ≤ 0.15
 (ชัดเจน)    (กำกวม)
   │         │
   ▼         ▼
 ⚡ Skip     🔬 Rerank
 ~15ms      ~300ms
```

### ผลทดสอบจริง (Benchmark)

| Query | Gap | Mode | เวลา | ทำไม? |
|-------|-----|------|------|-------|
| "Atomic Habits สอนอะไร..." | 0.032 | 🔬 Rerank | 0.576s | หลาย chunk ของเล่มเดียวกันแข่งกัน |
| "Rich Dad Poor Dad..." | 0.248 | ⚡ Skip | **0.020s** | ชัดเจน → ข้ามได้ |
| "วิธีเจรจาต่อรอง" | 0.428 | ⚡ Skip | **0.015s** | Top-1 ทิ้งห่างมาก |
| "วิธีฝึกสมาธิ..." | 0.092 | 🔬 Rerank | 0.190s | หลายหนังสือเกี่ยวข้องพอๆ กัน |
| "ซุนวูสอนอะไร..." | 0.175 | ⚡ Skip | **0.015s** | ซุนวูชัดเจน |

**ผลสรุป:** 3 ใน 5 queries ข้าม Reranker ได้ = ประหยัด GPU ~60% โดยไม่เสียความแม่นยำ

> 💡 **Tuning Tip:**
> - `RERANK_SCORE_GAP = 0.10` → rerank บ่อยขึ้น (แม่นกว่า แต่ช้ากว่า)
> - `RERANK_SCORE_GAP = 0.20` → skip บ่อยขึ้น (เร็วกว่า แต่เสี่ยงกว่า)
> - `RERANK_SCORE_GAP = 0.00` → rerank ทุกครั้ง (ปิด Adaptive)

**ไฟล์ที่เกี่ยวข้อง:** `rag_searcher.py` → `_should_rerank()`, `search()`

**ค่าตั้ง:** `config.py` → `RERANK_SCORE_GAP = 0.15`

---

## 9. Search Pipeline แบบเต็ม

### End-to-End Flow

```
User: "Atomic Habits สอนวิธีสร้างนิสัยอย่างไร"
│
▼ Stage 1a: Dense Search (GPU)
│  query → e5-large → query_vector
│  query_vector × FAISS index → Top 10 (by cosine similarity)
│  ⏱️ ~5ms
│
▼ Stage 1b: BM25 Search (CPU)
│  query → tokenize → ["atomic", "habits", "สอน", "วิธี", "สร้าง", "นิสัย"]
│  BM25.get_scores(tokens) → Top 10 (by TF-IDF)
│  ⏱️ ~1ms
│
▼ Stage 2: Score Merge
│  Dense scores: normalize to 0-1, weight × 0.7
│  BM25 scores:  normalize to 0-1, weight × 0.3
│  merged = dense + bm25 → Top 10 unique candidates
│  ⏱️ ~0.1ms
│
▼ Stage 3: Adaptive Reranking Decision
│  gap = Top-1 score - Top-2 score
│  ┌──────────────────────────┐
│  │ gap > 0.15 → ⚡ Skip!   │  ← Fast mode (~15ms total)
│  │ gap ≤ 0.15 → 🔬 Rerank │  ← Precision mode (~300ms)
│  └──────────────────────────┘
│  If Reranking: 10 pairs → bge-reranker-v2-m3.predict()
│  ⏱️ ~0ms (skip) or ~300ms (rerank)
│
▼ Output
│  [1] (Score: 0.99) [Atomic Habits] สรุป: กุญแจสู่การเปลี่ยนแปลง...
│  [2] (Score: 0.98) [Atomic Habits] เกริ่นนำ: พลังของนิสัยอะตอม...
│  ...
│
⏱️ Total: ~15ms (clear) or ~300-500ms (ambiguous)
```

---

## 10. โครงสร้างไฟล์และหน้าที่

```
RAG/
│
├── config.py              ⚙️ Central Configuration
│   └── ทุก setting อยู่ที่นี่: paths, models, chunking, hybrid weights
│
├── rag_creator.py         🔨 Index Builder (Core)
│   ├── TextChunker        → แบ่ง chunk + overlap
│   ├── tokenize_thai()    → tokenize สำหรับ BM25
│   └── RAGCreator         → อ่านข้อมูล → chunk → embed → save
│
├── rag_searcher.py        🔍 Search Engine (Core)
│   ├── tokenize_thai()    → tokenize query สำหรับ BM25
│   └── RAGSearcher        → Dense + BM25 → merge → adaptive rerank
│       ├── _dense_search()
│       ├── _bm25_search()
│       ├── _normalize_scores()
│       ├── _hybrid_merge()
│       ├── _should_rerank()  → ตัดสินใจ rerank หรือ skip
│       └── search()       → orchestrate ทั้งหมด
│
├── build_index.py         ▶️ Entry Point: สร้าง/สร้างใหม่ index
├── search.py              ▶️ Entry Point: ค้นหา interactive
├── test_rag.py            ▶️ Entry Point: ทดสอบชุดคำถาม
│
├── data/                  📂 ข้อมูลต้นทาง (.jsonl)
├── storage/               💾 Index + Data ที่สร้างแล้ว
│   ├── RAG_system.faiss       → FAISS vector index
│   ├── RAG_system_data.pkl    → ข้อความ chunk ต้นฉบับ
│   └── RAG_system_bm25.pkl   → Tokenized corpus สำหรับ BM25
│
└── docs/                  📖 เอกสาร
    └── technical_guide.md     → เอกสารฉบับนี้
```

### ความสัมพันธ์ระหว่างไฟล์

```
config.py ◄──────── rag_creator.py
    ▲                    ▲
    │                    │
    ├──────── rag_searcher.py
    │                    ▲
    │                    │
    ├──── build_index.py ┘ (ใช้ RAGCreator)
    ├──── search.py ───────── (ใช้ RAGSearcher)
    └──── test_rag.py ─────── (ใช้ RAGSearcher)
```

**หลักการออกแบบ:**
- **Separation of Concerns:** แยก Creator (สร้าง) กับ Searcher (ค้น) ชัดเจน
- **Single Source of Truth:** config.py เป็นที่เดียวที่เก็บ settings
- **Entry Points แยก:** แต่ละ script ทำงานเดียว (SRP)

---

## 11. Configuration — การตั้งค่า

ทุก setting อยู่ใน `config.py` — แก้ที่เดียว มีผลทั้งระบบ:

```python
# === Paths ===
BASE_DIR     = "/home/mikedev/RAG"
DATA_DIR     = "/home/mikedev/RAG/data"
STORAGE_DIR  = "/home/mikedev/RAG/storage"

# === Models ===
MODEL_EMBEDDING = ".../intfloat-multilingual-e5-large"   # Bi-Encoder
MODEL_RERANKER  = ".../BAAI-bge-reranker-v2-m3"          # Cross-Encoder

# === Index ===
INDEX_NAME = "RAG_system"

# === Chunking ===
CHUNK_SIZE    = 500    # ตัวอักษรต่อ chunk
CHUNK_OVERLAP = 100    # ส่วนซ้ำระหว่าง chunk

# === Hybrid Search ===
HYBRID_DENSE_WEIGHT = 0.7    # น้ำหนัก Dense (semantic)
HYBRID_BM25_WEIGHT  = 0.3    # น้ำหนัก BM25 (keyword)

# === Adaptive Reranking ===
RERANK_SCORE_GAP = 0.15      # Gap threshold (0.0=always rerank, 1.0=never)

# === Search ===
TOP_K_RETRIEVAL = 10   # จำนวน candidates จาก Stage 1
TOP_K_DISPLAY   = 5    # จำนวนผลลัพธ์สุดท้าย (หลัง Rerank)
BATCH_SIZE      = 32   # Batch size สำหรับ embedding
```

### คำแนะนำการปรับค่า

| Parameter | ปรับเมื่อ | คำแนะนำ |
|-----------|----------|---------|
| `CHUNK_SIZE` | ข้อมูลเป็นประโยคสั้นๆ เยอะ | ลดเป็น 300 |
| `CHUNK_SIZE` | ข้อมูลเป็นบทความยาว | เพิ่มเป็น 800 |
| `CHUNK_OVERLAP` | ค้นเจอข้อมูล "ครึ่งๆ กลางๆ" | เพิ่มเป็น 150 |
| `HYBRID_BM25_WEIGHT` | ค้นชื่อเฉพาะไม่ค่อยเจอ | เพิ่มเป็น 0.4 |
| `RERANK_SCORE_GAP` | อยากให้ rerank บ่อยขึ้น | ลดเป็น 0.10 |
| `RERANK_SCORE_GAP` | อยากให้ skip บ่อยขึ้น | เพิ่มเป็น 0.20 |
| `TOP_K_RETRIEVAL` | อยากให้ Reranker มี pool มากขึ้น | เพิ่มเป็น 20 |
| `BATCH_SIZE` | VRAM ไม่พอตอน build | ลดเป็น 16 |

---

## 12. ทรัพยากรที่ใช้ (VRAM/RAM)

### VRAM Budget (RTX 4060 — 8 GB)

| Component | VRAM | หมายเหตุ |
|-----------|------|---------|
| e5-large | ~2.2 GB | GPU (Embedding) |
| bge-reranker-v2-m3 | ~2.2 GB | GPU (Reranking) |
| FAISS Index | ~0.01 GB | GPU (Vector Search) |
| CUDA overhead | ~0.8 GB | GPU (Base) |
| **BM25** | **0 GB** | **CPU only!** |
| **รวม** | **~5.2 GB** | เหลือ ~2.8 GB |

### RAM Usage

| Component | RAM |
|-----------|-----|
| BM25 index | ~50 MB |
| Text data (pkl) | ~5 MB |
| Python + libs | ~500 MB |
| **รวม** | **~555 MB** |

### Storage

| File | ขนาด |
|------|------|
| RAG_system.faiss | ~23.5 MB |
| RAG_system_data.pkl | ~5.3 MB |
| RAG_system_bm25.pkl | ~1 MB |
| **รวม** | **~30 MB** |

---

## 13. ข้อมูลโมเดล AI

### Embedding: intfloat/multilingual-e5-large

- **ผู้พัฒนา:** Microsoft Research
- **สถาปัตยกรรม:** XLM-RoBERTa Large
- **วิธีฝึก:** Contrastive learning บนข้อมูลหลายภาษา
- **จุดเด่น:** ประสิทธิภาพสูงมากในทุกภาษา โดยเฉพาะ cross-lingual retrieval
- **Prefix:** ต้องใส่ `"passage: "` สำหรับเอกสาร และ `"query: "` สำหรับคำถาม
- **ที่เก็บ:** `/home/mikedev/MyModels/Model-RAG/intfloat-multilingual-e5-large`

### Reranker: BAAI/bge-reranker-v2-m3

- **ผู้พัฒนา:** Beijing Academy of AI (BAAI)
- **สถาปัตยกรรม:** XLM-RoBERTa (Cross-Encoder)
- **Version:** v2-m3 (Multi-lingual, Multi-granularity, Multi-function)
- **จุดเด่น:** ออกแบบมาเพื่อ multilingual reranking โดยเฉพาะ
- **ชนิด output:** Single float score (ยิ่งสูง ยิ่งเกี่ยวข้อง)
- **ที่เก็บ:** `/home/mikedev/MyModels/Model-RAG/BAAI-bge-reranker-v2-m3`

---

## 14. Flow Chart — ภาพรวมทุกขั้นตอน

### A. Index Building (ทำครั้งเดียว)

```
data/*.jsonl
    │
    ▼  อ่านไฟล์ + parse JSON
    │
Raw Documents (3,002 entries)
    │
    ▼  TextChunker (500 chars, overlap 100)
    │
Chunked Documents (5,738 chunks)
    │
    ├──────────────────┐
    ▼                  ▼
e5-large Encode     tokenize_thai()
    │                  │
    ▼                  ▼
FAISS Index         BM25 Corpus
(5,738 vectors)     (5,738 token lists)
    │                  │
    ▼                  ▼
.faiss file         _bm25.pkl file
    │
    ▼
_data.pkl file (original text)
```

### B. Full RAG Pipeline (ทุกครั้งที่ถาม)

```
User Query
    │
    ▼
┌──────────────────────────┐
│  Stage 0: HyDE Transform │
│  (Groq LLaMA 3.3 70B)   │
│  ~1.5s                   │
└────────────┬─────────────┘
             ▼
    ┌────────┴────────┐
    ▼                 ▼
Dense Search       BM25 Search
(FAISS + GPU)      (CPU)
~5ms               ~1ms
    │                 │
    ▼                 ▼
Normalize 0-1      Normalize 0-1
    │                 │
    └─────────┬───────┘
              ▼
    Weighted Merge (0.7 + 0.3)
              │
              ▼
    Top-10 Candidates
              │
              ▼
    Adaptive Reranking Decision
    ┌──────────────────────────┐
    │ gap > 0.15 → ⚡ Skip!   │
    │ gap ≤ 0.15 → 🔬 Rerank │
    └──────────────────────────┘
              │
              ▼
    Top-5 Final Results
              │
              ▼
┌──────────────────────────┐
│  Stage 3: LLM Generation │
│  (Gemini 2.5 Flash)      │
│  SSE Streaming → Web UI  │
│  ~5-8s                   │
└──────────────────────────┘
              │
              ▼
    🎯 Answer + Sources
    ⏱️ Total: ~7-10s (with HyDE + Gen)
```

---

## 10. HyDE — Query Transform

### HyDE คืออะไร?

**HyDE (Hypothetical Document Embedding)** คือเทคนิคที่ใช้ LLM สร้าง "เอกสารสมมติ" จากคำถาม แล้วใช้เอกสารสมมตินั้นเป็นคำค้นแทนคำถามเดิม

**ปัญหาที่ HyDE แก้:**
- ผู้ใช้ถามว่า "วิธีตื่นเช้า" → เอกสารจริงเขียนว่า "การปรับ circadian rhythm"
- คำถามกับเอกสารใช้คำคนละชุด → Dense Search หาไม่เจอ
- HyDE สร้างคำตอบสมมติที่มีคำศัพท์เหมือนเอกสาร → ค้นเจอ!

### ไฟล์: `core/query_transformer.py`

| Component | รายละเอียด |
|-----------|----------|
| LLM | Groq LLaMA 3.3 70B |
| Prompt | Concept-driven, เขียนเหมือนผู้เขียนหนังสือ |
| Max Tokens | 512 |
| Temperature | 0.7 (สร้างสรรค์เล็กน้อย) |
| Fallback | ถ้า API ล่ม → ใช้ query เดิม |
| เปิด/ปิด | `config.ENABLE_HYDE = True/False` |

### Flow

```
"วิธีสร้างนิสัยที่ดี"
    │
    ▼ (Groq LLaMA 3.3 70B)
    │
"การสร้างนิสัยที่ดีต้องเริ่มจากการเข้าใจกลไกของนิสัย
ซึ่งประกอบด้วย 4 ขั้นตอน: สัญญาณกระตุ้น (Cue),
แรงปรารถนา (Craving), การตอบสนอง (Response),
และรางวัล (Reward)..."
    │
    ▼ (ใช้เป็นคำค้นแทน)
    │
Hybrid Search → Adaptive Reranking → Results
```

---

## 11. LLM Generation — Gemini

### หน้าที่

รับผลลัพธ์จาก Search + Reranking แล้วสร้างคำตอบให้ผู้ใช้ พร้อมอ้างอิงแหล่งที่มา

### ไฟล์: `core/llm_generator.py`

| Component | รายละเอียด |
|-----------|----------|
| Model | Gemini 2.5 Flash |
| Temperature | 0.3 (ตอบชัด ไม่เพ้อ) |
| Max Tokens | ไม่จำกัด (ให้โมเดลตัดสินเอง) |
| System Prompt | Adaptive role — วิเคราะห์คำถาม → เลือกบทบาท |
| Streaming | SSE (Server-Sent Events) real-time |
| Key Rotation | Round-robin 10 keys |

### System Prompt Design

Prompt ถูกออกแบบให้ AI **วิเคราะห์คำถามก่อน** แล้วเลือกบทบาทที่เหมาะสม:

```
คำถามเรียบง่าย → ตอบสั้นตรงประเด็น
คำถามซับซ้อน → ใช้โครงสร้าง (ขั้นตอน, กรอบวิเคราะห์)
การวิเคราะห์ → ทำหน้าที่เป็นนักวิเคราะห์
การวางแผน → ทำหน้าที่เป็นนักวางแผน
คำถามทั่วไป → ทำหน้าที่เป็นที่ปรึกษา
```

### Pipeline ใน generate()

```
query + search_results
    │
    ▼ _build_context()
    │ → จัดรูปแบบ [แหล่งที่ 1] (ความเกี่ยวข้อง: 0.95)...
    │
    ▼ _build_prompt()
    │ → รวม query + context → prompt
    │
    ▼ Gemini API (stream=True)
    │ → yield chunks ทีละ token
    │
    ▼ SSE → Frontend
```

---

## 12. Core Package — API Keys & Modules

โฟลเดอร์ `core/` เก็บ modules ที่เกี่ยวกับ LLM, API keys, และ secrets

### โครงสร้าง

```
core/
├── __init__.py             # Package init
├── config.py               # .env loader → Settings singleton
├── key_manager.py          # Round-robin API key rotation
├── llm_generator.py        # Gemini generation (sync + streaming)
└── query_transformer.py    # HyDE + Query Rewriting (Groq)
```

### Config แยก 2 ระดับ (Separation of Concerns)

| ไฟล์ | หน้าที่ | ตัวอย่าง |
|------|--------|--------|
| `config.py` (root) | RAG tuning parameters | CHUNK_SIZE, HYBRID_WEIGHT, RERANK_GAP |
| `core/config.py` | API keys & LLM settings | GEMINI_API_KEYS, TEMPERATURE |

### Key Manager — Round-Robin Rotation

```
Key Pool: [K1, K2, K3, K4, K5, K6, K7, K8, K9, K10]
                                                 │
Request 1 → K1                                   │
Request 2 → K2                                   │
Request 3 → K3 ... → Request 10 → K10 → K1 (วนซ้ำ)
```

**ทำไมต้อง rotate?**
- Google Gemini มี rate limit ต่อ key (~15 req/min)
- 10 keys → รองรับ 150 req/min
- ถ้า key หมด → `get_key()` return None → แสดง error

---

## 13. Web UI — FastAPI + SSE

### ไฟล์: `web_server.py` + `web/`

Web UI เป็น chat interface แบบ dark theme ที่แสดงผลแบบ real-time streaming

### Technical Stack

| Component | Technology |
|-----------|----------|
| Backend | FastAPI + uvicorn |
| Streaming | SSE (Server-Sent Events) via sse-starlette |
| Frontend | Vanilla HTML/CSS/JS |
| Markdown | marked.js |
| Design | Dark theme + glassmorphism + Inter font |

### SSE Event Types

| Event | Data | จังหวะ |
|-------|------|--------|
| `status` | `{stage, message}` | เมื่อเริ่มแต่ละ stage |
| `hyde` | `{hyde_query, time}` | หลัง HyDE สำเร็จ |
| `sources` | `{sources[], search_time}` | หลัง Search เสร็จ |
| `token` | `{text}` | ทุก token ที่ Gemini สร้าง |
| `done` | `{hyde_time, search_time, gen_time, total_time}` | จบ pipeline |

### API Endpoint

```
POST /api/ask
Body: { "query": "...", "use_hyde": true }
Response: SSE stream (event: status → hyde → sources → token* → done)
```

### Frontend Features

- 🌙 Dark theme + purple accent palette
- 💬 Chat bubbles (user + AI)
- 📑 Sources panel (คะแนน + เนื้อหา)
- ⏱️ Timing bar (HyDE / Search / Gen / Total)
- 🪄 HyDE toggle switch
- 💡 Suggestion chips
- 📱 Responsive design

---

## 14. โครงสร้างไฟล์และหน้าที่

```
RAG/
├── config.py               # ⚙️  Central config (paths, models, tuning)
├── rag_creator.py          # 🔨 Chunking + embedding + index building
├── rag_searcher.py         # 🔍 Hybrid search + adaptive reranking
├── build_index.py          # ▶️  CLI: build/rebuild index
├── search.py               # ▶️  CLI: interactive search (retrieval only)
├── ask.py                  # 🤖 CLI: full RAG pipeline
├── web_server.py           # 🌐 FastAPI + SSE streaming server
├── test_rag.py             # ✅ Test suite
│
├── core/                   # 📦 Core modules
│   ├── config.py           #   🔐 .env loader (API keys)
│   ├── key_manager.py      #   🔑 Round-robin key rotation
│   ├── llm_generator.py    #   🤖 Gemini generation
│   └── query_transformer.py#   🪄 HyDE + Query Rewriting
│
├── web/                    # 🎨 Frontend
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── data/                   # 📂 Source .jsonl files
├── storage/                # 💾 FAISS + BM25 indices
└── .env                    # 🔐 API keys
```

---

## 15. Configuration — การตั้งค่า

ระบบแยก config เป็น 2 ระดับ:

### `config.py` (root) — RAG Tuning

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
```

### `core/config.py` — API Keys & LLM Settings

```python
# Environment Variables (via .env file)
GEMINI_API_KEYS     # Comma-separated Gemini API keys (10 keys)
GROQ_API_KEYS       # Comma-separated Groq API keys (3 keys)
GEMINI_MODEL        # Default: gemini-2.5-flash
GEMINI_TEMPERATURE  # Default: 0.3
GROQ_MODEL          # Default: llama-3.3-70b-versatile
GROQ_TEMPERATURE    # Default: 0.7
```

---

> 📅 **Last Updated:** February 2025
> 📝 **Author:** Auto-generated documentation
> 🔖 **Version:** 2.0 — Full RAG Pipeline (HyDE + Hybrid Search + Adaptive Reranking + Gemini Generation + Web UI)
