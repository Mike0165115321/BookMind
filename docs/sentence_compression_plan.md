# Sentence-Level Context Compression
## แผนการพัฒนา BookMind RAG System
**เวอร์ชัน:** 4.0 — Sentence-Level Compression  
**วันที่ร่าง:** May 2026  
**ผู้ร่าง:** Aetox.dev

---

## ภาพรวมปัญหา

ระบบปัจจุบันส่ง chunk ดิบทั้งก้อนให้ generator ทุกครั้ง ไม่ว่าคำถามจะง่ายหรือซับซ้อน ผลคือโมเดลได้รับ context มากกว่าที่ต้องการ และตอบยาวเกินความจำเป็น

**ต้นเหตุจริง:** ปัญหาไม่ได้อยู่ที่โมเดล แต่อยู่ที่ input — ท่อน้ำไม่สะอาดพอ โมเดลจึงกรองเองไม่ถูก

**แนวทางแก้:** แทนที่จะสั่งโมเดลผ่าน prompt ให้ทำ compression ก่อนถึงโมเดล โดยใช้ infrastructure ที่มีอยู่แล้วทั้งหมด ไม่เพิ่ม API call ใดๆ

---

## สถาปัตยกรรมใหม่

### Pipeline ปัจจุบัน

```
query
  → HyDE (Groq)
  → Hybrid Search (FAISS + BM25)
  → Adaptive Reranker (Cross-Encoder)
  → _build_context()  ← ส่ง chunk ดิบทั้งก้อน
  → Generator (Gemini)
```

### Pipeline ใหม่

```
query
  → HyDE (Groq)
  → Hybrid Search (FAISS + BM25)
  → Adaptive Reranker (Cross-Encoder)  ← chunk level เหมือนเดิม
  → [NEW] Sentence Splitter
  → [NEW] Embedding Filter             ← กรองหยาบ ตัดประโยคไม่เกี่ยว
  → [NEW] Reranker Score               ← เรียงละเอียด ใช้ model ที่มีอยู่แล้ว
  → [NEW] Top-N Selector               ← เลือกแค่ประโยคที่ตอบตรง
  → _build_context()  ← ส่ง compressed context
  → Generator (Gemini)
```

**หลักการสำคัญ:** Embedding Filter และ Reranker ไม่ได้เพิ่ม model ใหม่ ใช้ model เดิมที่โหลดอยู่ใน GPU memory แล้ว

---

## สองขั้นตอนหลัก

### ขั้นที่ 1 — Embedding Filter (กรองหยาบ)

**หน้าที่:** ตัดประโยคที่ความหมายไม่เกี่ยวกับ query ออก

**วิธีทำงาน:**
1. ตัด chunk แต่ละอันออกเป็นประโยค
2. Encode ทุกประโยคด้วย e5-large ที่โหลดอยู่แล้ว
3. คำนวณ cosine similarity กับ query vector
4. ตัดประโยคที่ similarity ต่ำกว่า threshold ออก

**ทำไมต้องทำก่อน:**
- เร็ว เพราะใช้ dot product ล้วนๆ ไม่มี cross-attention
- ลดจำนวนประโยคก่อนส่งให้ reranker ทำให้ reranker วิ่งบน corpus เล็กลง
- ไม่เพิ่ม latency มากนักเพราะ model warm อยู่แล้ว

**ตัวอย่าง:**
```
chunk: "การสร้างนิสัยต้องใช้เวลา / James Clear แนะนำ 2 นาที rule /
        หนังสือขายดีทั่วโลก / แปลแล้วกว่า 50 ภาษา"

query: "Atomic Habits แนะนำให้เริ่มต้นยังไง"

หลัง embedding filter:
✓ "James Clear แนะนำ 2 นาที rule"       similarity: 0.82
✗ "หนังสือขายดีทั่วโลก"                  similarity: 0.21  ← ตัดออก
✗ "แปลแล้วกว่า 50 ภาษา"                 similarity: 0.18  ← ตัดออก
✓ "การสร้างนิสัยต้องใช้เวลา"             similarity: 0.61
```

---

### ขั้นที่ 2 — Reranker Score (เรียงละเอียด)

**หน้าที่:** จัดอันดับประโยคที่เหลือว่าอันไหนตอบคำถามนี้ได้ตรงที่สุด

**วิธีทำงาน:**
1. รับประโยคที่ผ่าน embedding filter มา
2. ใช้ bge-reranker-v2-m3 ที่โหลดอยู่แล้ว score ทุก pair (query, sentence)
3. เรียงตาม reranker score
4. เลือก top-N ประโยค

**ทำไมต้องทำหลัง embedding:**
- Cross-encoder ช้ากว่า bi-encoder มาก
- ถ้าวิ่งบนทุกประโยคก่อน filter จะช้าเกินไป
- วิ่งบน corpus ที่ filter แล้ว latency เพิ่มน้อยมาก

**ความต่างจาก embedding filter:**

| | Embedding Filter | Reranker Score |
|---|---|---|
| มองที่ | ความหมายใกล้เคียง | ตอบคำถามได้จริงไหม |
| เร็ว/ช้า | เร็ว | ช้ากว่า แต่แม่นกว่า |
| ตัวอย่าง | "นิสัย" กับ "habit" ใกล้กัน | "ทำ 2 นาทีก่อน" ตอบ "เริ่มยังไง" ได้จริง |

---

## จำนวนประโยคที่ส่ง (Top-N)

**ปัญหาของ fixed N:** คำถามสั้นกับซับซ้อนควรได้ context ต่างกัน

**แนวทาง:** ใช้ dynamic N โดยให้ decomposer ที่มีอยู่แล้วส่ง signal มา

```
DecompositionResult เพิ่ม field:
  query_type: "simple" / "complex"

simple  → top 3-5 ประโยค
complex → top 8-12 ประโยค
```

ไม่ต้องสร้างระบบใหม่ แค่เพิ่ม field เดียวใน dataclass ที่มีอยู่

---

## ไฟล์ที่ต้องแก้

### ไฟล์ใหม่ที่ต้องสร้าง

```
core/retrieval/compressor.py   ← logic ทั้งหมดอยู่ที่นี่
```

ประกอบด้วย:
- `SentenceSplitter` — ตัดข้อความเป็นประโยค รองรับไทย+อังกฤษ
- `EmbeddingFilter` — กรองด้วย cosine similarity
- `SentenceReranker` — score ด้วย cross-encoder
- `SentenceCompressor` — orchestrate ทั้งสามขั้นตอน

### ไฟล์ที่ต้องแก้

| ไฟล์ | สิ่งที่แก้ |
|---|---|
| `core/retrieval/pipeline.py` | เรียก SentenceCompressor หลัง rerank |
| `core/llm/generator.py` | `_build_context()` รับ compressed sentences แทน chunks |
| `core/agentic/types.py` | เพิ่ม `context_budget` ใน DecompositionResult |
| `core/query_decomposer.py` | ให้ decomposer return `context_budget` ด้วย |
| `config.py` | เพิ่ม constants ใหม่ |

---

## Constants ใหม่ใน config.py

```python
# Sentence Compression
COMPRESSION_ENABLED = True
COMPRESSION_EMBEDDING_THRESHOLD = 0.45  # ต่ำกว่านี้ตัดออก
COMPRESSION_TOP_N_SIMPLE = 5            # คำถามง่าย
COMPRESSION_TOP_N_COMPLEX = 12          # คำถามซับซ้อน
COMPRESSION_MIN_SENTENCE_LENGTH = 10    # ตัดประโยคสั้นเกินออก
```

---

## สิ่งที่ไม่ต้องทำ

- ไม่ต้อง rebuild index
- ไม่ต้องเพิ่ม API key หรือ provider ใหม่
- ไม่ต้องโหลด model ใหม่
- ไม่ต้องแก้ frontend เลย
- ไม่ต้องแก้ system prompt

---

## Latency ที่คาดว่าจะเพิ่ม

| ขั้นตอน | เวลาเพิ่ม (ประมาณ) | หมายเหตุ |
|---|---|---|
| Sentence splitting | ~1ms | regex ล้วนๆ |
| Embedding filter | ~10-20ms | model warm อยู่แล้ว |
| Reranker score | ~50-100ms | วิ่งบน corpus เล็ก |
| **รวม** | **~60-120ms** | แลกกับ context ที่สะอาดขึ้น |

เทียบกับ HyDE ที่ใช้เวลา ~1500ms — overhead นี้เล็กมากครับ

---

## ลำดับการพัฒนาที่แนะนำ

```
Step 1: สร้าง SentenceSplitter + ทดสอบกับข้อมูลจริง
         → ตรวจว่าตัดประโยคไทยได้ถูกต้อง

Step 2: เพิ่ม EmbeddingFilter + กำหนด threshold
         → ทดสอบว่า threshold 0.45 เหมาะกับ knowledge base นี้ไหม

Step 3: เพิ่ม SentenceReranker
         → วัด latency จริงก่อน deploy

Step 4: เพิ่ม context_budget ใน DecompositionResult
         → ทดสอบ simple vs complex query

Step 5: integrate เข้า pipeline.py และ generator.py
         → A/B test กับ pipeline เดิม
```

---

## วิธีวัดว่าดีขึ้นจริง

ก่อน deploy ควรมี baseline ครับ วัดสามอย่างนี้กับ query ชุดเดิม

1. **ความยาวคำตอบ** — ควรสั้นลงเมื่อ query ง่าย
2. **Token ที่ส่งให้ generator** — ควรลดลงเฉลี่ย 40-60%
3. **ความถูกต้องของคำตอบ** — ต้องไม่แย่ลง

ถ้าสามอย่างนี้ดีขึ้นหรือเท่าเดิม แปลว่า compression ทำงานถูกต้อง

---

*แผนนี้ใช้ infrastructure ที่มีอยู่ทั้งหมด ไม่มีต้นทุนเพิ่ม และสามารถ enable/disable ผ่าน `COMPRESSION_ENABLED` ใน config.py ได้ทันที*
