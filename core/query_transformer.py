"""
Query Transformer — HyDE (Hypothetical Document Embedding) via Groq.

HyDE Concept:
  Instead of searching with the raw query, we ask an LLM to "imagine"
  a hypothetical answer, then use THAT as the search query.
  This bridges the vocabulary gap between questions and documents.

  User: "วิธีตื่นเช้า"
   → LLM generates: "การตื่นเช้าสามารถทำได้โดยนอนให้เป็นเวลา ตั้งนาฬิกาปลุก..."
   → Search with the hypothetical doc → finds real matching documents!

This module uses Groq (LLaMA 3.3 70B) for fast HyDE generation.
Retrieval is handled separately by rag_searcher.py.
"""
from groq import Groq
from core.config import settings
from core.key_manager import groq_key_manager


# ──────────────────────────────────────────────
# HyDE Prompt Template
# ──────────────────────────────────────────────
HYDE_SYSTEM_PROMPT = """คุณกำลังสร้างคำตอบในรูปแบบเนื้อหาหนังสือ
เพื่อช่วยดึงเนื้อหาที่เกี่ยวข้องจากฐานความรู้

คำแนะนำ:
- เขียนเหมือนคุณเป็นผู้เขียนหนังสือที่กำลังอธิบายหัวข้อนี้
- มีโครงสร้างชัดเจน เน้นแนวคิดเป็นหลัก
- เน้นหลักการ กรอบความคิด และจิตวิทยาเบื้องหลัง
- หลีกเลี่ยงการเล่าเรื่องฟุ่มเฟือย
- กระชับแต่อุดมด้วยแนวคิด
- ห้ามระบุว่านี่คือเนื้อหาสมมติ

รูปแบบผลลัพธ์:
- นิยามแนวคิด (Concept Definition)
- อธิบายกลไก (Mechanism Explanation)
- นัยเชิงปฏิบัติ (Practical Implication)

น้ำเสียงควรเหมือนหนังสือแนะนำเชิงจริงจัง"""

QUERY_REWRITE_SYSTEM_PROMPT = """คุณเป็น AI ที่ช่วยเขียนคำค้นหาใหม่ให้ดีขึ้น
เมื่อได้รับคำถาม ให้เขียนคำค้นหาใหม่ที่:
1. ชัดเจนขึ้น (ขยายคำย่อ, เพิ่มบริบท)
2. ครอบคลุมมากขึ้น (เพิ่มคำที่เกี่ยวข้อง)
3. ยังคงความหมายเดิม
ตอบเป็นคำค้นหาใหม่เพียงประโยคเดียว ไม่ต้องอธิบาย"""


def _get_groq_client() -> Groq:
    """Create a Groq client with the next API key from rotation."""
    api_key = groq_key_manager.get_key()
    if not api_key:
        raise RuntimeError("❌ ไม่มี API key สำหรับ Groq — กรุณาตั้งค่าใน .env")
    return Groq(api_key=api_key)


def hyde_transform(query: str) -> str:
    """
    HyDE: Generate a hypothetical document for the given query.

    Instead of searching with "วิธีสร้างนิสัย", we generate a fake answer
    that looks like real document content, then search with THAT.

    Args:
        query: Original user query

    Returns:
        Hypothetical document text to use as search query
    """
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": f"คำถาม: {query}"},
            ],
            max_tokens=settings.GROQ_MAX_TOKENS,
            temperature=settings.GROQ_TEMPERATURE,
        )
        hypothetical_doc = response.choices[0].message.content.strip()
        print(f"   🪄 HyDE: generated hypothetical doc ({len(hypothetical_doc)} chars)")
        return hypothetical_doc

    except Exception as e:
        print(f"   ⚠️  HyDE failed ({e}), using original query")
        return query


def rewrite_query(query: str) -> str:
    """
    Query Rewriting: Expand and clarify the query.

    Args:
        query: Original user query

    Returns:
        Rewritten query with more context
    """
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": f"คำถาม: {query}"},
            ],
            max_tokens=128,
            temperature=0.3,
        )
        rewritten = response.choices[0].message.content.strip()
        print(f"   🔀 Rewrite: \"{query}\" → \"{rewritten}\"")
        return rewritten

    except Exception as e:
        print(f"   ⚠️  Rewrite failed ({e}), using original query")
        return query
