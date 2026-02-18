"""
LLM Generator — Generates answers from retrieved context using Gemini.

Responsibilities:
  - Build prompt from query + retrieved chunks
  - Call Gemini API with key rotation
  - Return structured response with answer + sources

This module ONLY handles generation. Retrieval is handled by rag_searcher.py.
"""
from google import genai
from google.genai import types
from core.config import settings
from core.key_manager import gemini_key_manager


# ──────────────────────────────────────────────
# System Prompt Template
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """
คุณเป็น AI ที่ใช้ความรู้จากหนังสือที่ดึงมาเป็นฐานในการตอบคำถาม

ภารกิจหลัก:
- ใช้ข้อมูลจากหนังสือเป็นแหล่งความรู้หลัก
- ห้ามแต่งเนื้อหาหนังสือขึ้นเอง
- หากหนังสือไม่ได้กล่าวถึงโดยตรง ให้ระบุชัดเจน และเสนอแนวคิดใกล้เคียงอย่างมีเหตุผล

กระบวนการก่อนตอบ (ให้คิดภายใน ไม่ต้องแสดงขั้นตอนนี้):
1. วิเคราะห์ว่า ผู้ใช้ต้องการอะไร
   - คำอธิบาย
   - การวิเคราะห์
   - การวางแผน
   - การออกแบบโครงสร้าง
   - การให้คำปรึกษา
   - การแก้ปัญหาเชิงกลยุทธ์
2. เลือกบทบาทที่เหมาะสมกับคำขอ เช่น
   - หากเป็นการวิเคราะห์ → ทำหน้าที่เป็นนักวิเคราะห์
   - หากเป็นการวางแผน → ทำหน้าที่เป็นนักวางแผน
   - หากเป็นการออกแบบ → ทำหน้าที่เป็นผู้ออกแบบกรอบแนวคิด
   - หากเป็นคำถามทั่วไป → ทำหน้าที่เป็นที่ปรึกษา
3. เลือกความลึกและรูปแบบคำตอบให้เหมาะกับความซับซ้อนของคำถาม

หลักการตอบ:
- ตอบให้กระชับเท่าที่จำเป็น
- หากคำถามเรียบง่าย → ตอบสั้นและตรงประเด็น
- หากคำถามซับซ้อน → สามารถใช้โครงสร้าง เช่น ขั้นตอน ลำดับ แผนงาน หรือกรอบวิเคราะห์
- ใช้โครงสร้างเฉพาะเมื่อช่วยให้เข้าใจง่ายขึ้น ไม่ต้องฝืนแบ่งหัวข้อเสมอ

การอ้างอิง:
- เมื่ออ้างอิงหนังสือ ให้ใช้รูปแบบ
  "ในหนังสือ [ชื่อหนังสือ] กล่าวถึงว่า..."
- ห้ามสร้างเลขหน้าหรือคำพูดตรง หากไม่มีในข้อมูล

กรณีข้อมูลไม่เพียงพอ:
ให้ตอบว่า
"ในเอกสารที่มีอยู่ ไม่ได้กล่าวถึงประเด็นนี้โดยตรง..."
แล้วเสนอแนวคิดใกล้เคียงอย่างมีเหตุผลและกระชับ

เป้าหมายสูงสุด:
แปลงความรู้จากหนังสือ → ให้กลายเป็นคำตอบที่ใช้งานได้จริง
"""




def _build_context(search_results: list) -> str:
    """
    Build context string from search results.

    Args:
        search_results: List of (text, score) tuples from RAGSearcher.search()

    Returns:
        Formatted context string with numbered sources
    """
    if not search_results:
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"

    context_parts = []
    for i, (text, score) in enumerate(search_results, 1):
        context_parts.append(f"[แหล่งที่ {i}] (ความเกี่ยวข้อง: {score:.2f})\n{text}")

    return "\n\n---\n\n".join(context_parts)


def _build_prompt(query: str, context: str) -> str:
    """
    Build the user prompt combining query and retrieved context.

    Args:
        query: User's question
        context: Formatted context from search results

    Returns:
        Complete prompt string
    """
    return f"""คำถาม: {query}

ข้อมูลอ้างอิง:
{context}

จากข้อมูลอ้างอิงข้างต้น กรุณาตอบคำถามอย่างละเอียดและอ้างอิงแหล่งที่มา"""


def _get_client() -> genai.Client:
    """Create a Gemini client with the next API key from rotation."""
    api_key = gemini_key_manager.get_key()
    if not api_key:
        raise RuntimeError("❌ ไม่มี API key สำหรับ Gemini — กรุณาตั้งค่าใน .env")
    return genai.Client(api_key=api_key)


def generate(query: str, search_results: list, stream: bool = False) -> str:
    """
    Generate an answer using Gemini from retrieved search results.

    Args:
        query: User's question
        search_results: List of (text, score) tuples from RAGSearcher.search()
        stream: If True, yields chunks for streaming output

    Returns:
        Generated answer string (or generator if stream=True)
    """
    # 1. Build prompt
    context = _build_context(search_results)
    prompt = _build_prompt(query, context)

    # 2. Get client with rotated key
    client = _get_client()

    # 3. Generation config (no max_output_tokens — let the model decide)
    gen_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=settings.GEMINI_TEMPERATURE,
    )

    # 4. Generate
    if stream:
        return _stream_generate(client, prompt, gen_config)
    else:
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config=gen_config,
        )
        return response.text


def _stream_generate(client: genai.Client, prompt: str, gen_config):
    """Generator that yields text chunks for streaming output."""
    for chunk in client.models.generate_content_stream(
        model=settings.GEMINI_MODEL,
        contents=prompt,
        config=gen_config,
    ):
        if chunk.text:
            yield chunk.text
