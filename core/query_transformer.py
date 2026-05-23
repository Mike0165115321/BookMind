from core.llm.manager import llm_manager
from core.llm.shared.types import Message, ProviderName, GenerationConfig

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
- อธิบายกลกลไก (Mechanism Explanation)
- นัยเชิงปฏิบัติ (Practical Implication)

น้ำเสียงควรเหมือนหนังสือแนะนำเชิงจริงจัง"""

QUERY_REWRITE_SYSTEM_PROMPT = """คุณเป็น AI ที่ช่วยเขียนคำค้นหาใหม่ให้ดีขึ้น
เมื่อได้รับคำถาม ให้เขียนคำค้นหาใหม่ที่:
1. ชัดเจนขึ้น (ขยายคำย่อ, เพิ่มบริบท)
2. ครอบคลุมมากขึ้น (เพิ่มคำที่เกี่ยวข้อง)
3. ยังคงความหมายเดิม
ตอบเป็นคำค้นหาใหม่เพียงประโยคเดียว ไม่ต้องอธิบาย"""

def hyde_transform(query: str, provider: ProviderName = ProviderName.GROQ, model_name: str = None) -> str:
    """
    HyDE: Generate a hypothetical document via llm_manager.
    """
    try:
        messages = [
            Message(role="system", content=HYDE_SYSTEM_PROMPT),
            Message(role="user", content=f"คำถาม: {query}")
        ]
        
        # Use a higher temperature for creativity in HyDE
        config = GenerationConfig(temperature=0.7, max_tokens=512)
        
        response = llm_manager.generate(provider, messages, model_name=model_name, config=config)
        
        print(f"   🪄 HyDE: generated via {response.provider} ({response.model_name}) in {response.latency_ms:.2f}ms")
        return response.text

    except Exception as e:
        print(f"   ⚠️ HyDE failed ({e}), using original query")
        return query

def rewrite_query(query: str, provider: ProviderName = ProviderName.GROQ, model_name: str = None) -> str:
    """
    Query Rewriting: Expand and clarify the query via llm_manager.
    """
    try:
        messages = [
            Message(role="system", content=QUERY_REWRITE_SYSTEM_PROMPT),
            Message(role="user", content=f"คำถาม: {query}")
        ]
        
        config = GenerationConfig(temperature=0.3, max_tokens=128)
        
        response = llm_manager.generate(provider, messages, model_name=model_name, config=config)
        
        print(f"   🔀 Rewrite: \"{query}\" → \"{response.text}\" (via {response.provider})")
        return response.text

    except Exception as e:
        print(f"   ⚠️ Rewrite failed ({e}), using original query")
        return query

# ──────────────────────────────────────────────
# Web Search HyDE Settings & Prompts
# ──────────────────────────────────────────────
WEB_SEARCH_HYDE_PROMPT = """คุณเป็น AI ที่ช่วยสกัดคำค้นหาที่กระชับและส่งผลลัพธ์สูงสำหรับการค้นหาบนเสิร์ชเอนจิน (Search Engine Optimization)
หน้าที่ของคุณคือเปลี่ยนคำถามทักทายทั่วไปให้กลายเป็น "ย่อหน้าจำลองสั้นๆ (Hypothetical Web Document) ความยาวไม่เกิน 1-2 ประโยค" ที่น่าจะปรากฏอยู่บนหน้าเว็บเพจหรือข่าวสารเพื่อช่วยจับคู่คีย์เวิร์ด
- ห้ามเขียนเป็นบทความยาว
- ห้ามระบุคำทักทาย
- ห้ามเขียนคำอธิบายเพิ่มเติม
ตอบกลับด้วยข้อความจำลองสั้นๆ นั้นทันที"""

def web_hyde_transform(query: str, provider: ProviderName = ProviderName.GEMINI, model_name: str = None) -> str:
    """
    Web-Specific HyDE: Generate a short hypothetical web document / search-friendly query snippet.
    """
    try:
        messages = [
            Message(role="system", content=WEB_SEARCH_HYDE_PROMPT),
            Message(role="user", content=f"คำถาม: {query}")
        ]
        
        # Use lower temperature for speed and precision
        config = GenerationConfig(temperature=0.3, max_tokens=128)
        
        response = llm_manager.generate(provider, messages, model_name=model_name, config=config)
        
        print(f"   🌐 Web HyDE: generated via {response.provider} ({response.model_name}) in {response.latency_ms:.2f}ms")
        return response.text.strip()

    except Exception as e:
        print(f"   ⚠️ Web HyDE failed ({e}), using original query")
        return query

