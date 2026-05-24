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

WEB_SEARCH_HYDE_PROMPT = """คุณเป็นผู้เชี่ยวชาญระดับสูงในการเขียนคำค้นหาบน Search Engine (เช่น Google, DuckDuckGo)
หน้าที่ของคุณคือเปลี่ยนคำถามของผู้ใช้ ให้กลายเป็นคำค้นหา (Search Query) ที่สั้น กระชับ ตรงประเด็น และมีคีย์เวิร์ดสำคัญครบถ้วนเพื่อผลลัพธ์การค้นหาที่ดีที่สุด

กฎเหล็กสำหรับการตอบกลับ:
- ตอบกลับเฉพาะ คำค้นหาหลัก (Keywords) 2-5 คำ เว้นวรรคแยกคำด้วยช่องว่าง เท่านั้น
- ห้ามใส่เครื่องหมายคำพูด (Quotes) หรือสัญลักษณ์พิเศษใดๆ
- ห้ามเขียนอธิบาย ห้ามใส่หัวข้อ และห้ามเกริ่นนำใดๆ ทั้งสิ้น
- ห้ามใส่คำถามทักทายทั่วไป เช่น สวัสดีครับ, ช่วยหาหน่อย

ตัวอย่างการทำงาน:
- "สวัสดีครับช่วยหาประวัติของสตีฟ จอบส์หน่อยครับ" -> "ประวัติ สตีฟ จอบส์ Apple"
- "สอนวิธีเขียน React เชื่อมต่อ API แบบเข้าใจง่าย" -> "React เชื่อมต่อ API ตัวอย่างโค้ด"
- "ราคาหุ้น NVIDIA ตอนนี้ขึ้นหรือลงเพราะอะไร" -> "หุ้น NVIDIA วิเคราะห์ ราคาล่าสุด"

จงสกัดคำค้นหาจากคำถามของผู้ใช้อย่างเคร่งครัดตามกฎด้านบน:"""

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


# ──────────────────────────────────────────────
# Query Contextualization (Multi-turn Support)
# ──────────────────────────────────────────────
CONTEXT_TRIGGERS = [
    "เขา", "เธอ", "มัน", "นั้น", "ที่พูดถึง", "แบบนั้น", "คนนั้น", "ผู้นั้น", 
    "สิ่งนี้", "เหล่านี้", "ดังกล่าว", "คราวก่อน", "เมื่อกี้", "ต่อ", "เพิ่มเติม", "ยังไง", "อย่างไร",
    "แรก", "สุดท้าย", "อีก", "เล่ม", "ข้อ", "ตอน", "เรื่อง", "อัน", "ไหน", "ก่อน", "หลัง",
    "he", "she", "it", "they", "them", "him", "her", "that", "this", "these", "those"
]

def needs_contextualization(query: str) -> bool:
    q_lower = query.lower()
    return any(t in q_lower for t in CONTEXT_TRIGGERS)

CONTEXTUALIZE_SYSTEM_PROMPT = """คุณคือผู้ช่วย AI ที่เชี่ยวชาญในการปรับแต่งคำถาม (Query Contextualization)
หน้าที่ของคุณคือวิเคราะห์คำถามล่าสุดของผู้ใช้ ร่วมกับประวัติการสนทนาที่ผ่านมา

หากคำถามล่าสุดมีการอ้างอิงถึงหัวข้อหรือประโยคก่อนหน้า (เช่น การใช้คำสรรพนาม 'เขา', 'มัน', 'สิ่งนี้', 'พวกนั้น' หรือคำที่เกี่ยวเนื่องจากบริบทเดิม)
ให้คุณเขียนคำถามใหม่แบบเต็ม (Self-contained) ที่สมบูรณ์ ชัดเจน มีบริบทครบถ้วน โดยยังคงเจตนาเดิมของผู้ใช้ เพื่อนำไปใช้สืบค้นข้อมูลในคลังเอกสารต่อได้ทันที

กฎข้อบังคับ:
1. หากคำถามล่าสุดชัดเจนและสมบูรณ์ในตัวเองอยู่แล้ว หรือไม่ได้อ้างอิงถึงบริบทเก่า ให้ตอบกลับด้วยคำถามเดิมของผู้ใช้คำเดิมทุกประการ ห้ามเปลี่ยนแปลงอะไร
2. ตอบกลับเฉพาะตัวข้อความคำถามที่ปรับแต่งแล้วหรือคำถามเดิมเท่านั้น ห้ามเขียนอธิบาย ห้ามใส่หัวข้อ และห้ามเขียนเกริ่นนำใดๆ ทั้งสิ้นเด็ดขาด

ตัวอย่างการคุย:
ประวัติ:
User: ใครคือผู้ก่อตั้ง Apple?
AI: สตีฟ จอบส์ และ สตีฟ วอซเนียก
คำถามใหม่: แล้วเขาเกิดปีไหน?
-> ผลลัพธ์: สตีฟ จอบส์ เกิดปีอะไร

ประวัติ:
User: แนะนำหนังสือจิตวิทยาหน่อย
AI: มีหนังสือ Atomic Habits และ Psychology of Money ครับ
คำถามใหม่: เล่มแรกพูดเกี่ยวกับอะไร?
-> ผลลัพธ์: หนังสือ Atomic Habits พูดเกี่ยวกับอะไร"""

def contextualize_query(query: str, chat_history: list[dict], provider: ProviderName, model_name: str = None) -> str:
    """
    If chat_history is present and query contains pronouns or context references, 
    rewrite the query to be a self-contained query using the LLM.
    """
    if not chat_history or len(chat_history) == 0:
        return query
        
    if not needs_contextualization(query):
        print(f"   ⚡ Smart Skip: Query \"{query}\" does not require contextualization.")
        return query

    try:
        # Build history prompt structure for LLM
        history_str = []
        # Take the last 5 turns (max 10 messages)
        for msg in chat_history[-10:]:
            role_label = "User" if msg["role"] == "user" else "AI"
            history_str.append(f"{role_label}: {msg['content']}")
            
        history_context = "\n".join(history_str)
        
        prompt = f"""ประวัติการสนทนา:
{history_context}

---
คำถามล่าสุด: {query}"""

        messages = [
            Message(role="system", content=CONTEXTUALIZE_SYSTEM_PROMPT),
            Message(role="user", content=prompt)
        ]
        
        config = GenerationConfig(temperature=0.1, max_tokens=256)
        response = llm_manager.generate(provider, messages, model_name=model_name, config=config)
        
        rewritten = response.text.strip()
        # Clean up any potential markdown wrapper or prefixes if LLM hallucinated them
        if rewritten.startswith("-> ผลลัพธ์:"):
            rewritten = rewritten.replace("-> ผลลัพธ์:", "").strip()
        elif rewritten.startswith("ผลลัพธ์:"):
            rewritten = rewritten.replace("ผลลัพธ์:", "").strip()
            
        print(f"   🧠 Contextualized: \"{query}\" → \"{rewritten}\" (via {response.provider})")
        return rewritten

    except Exception as e:
        print(f"   ⚠️ Contextualization failed ({e}), using original query")
        return query


