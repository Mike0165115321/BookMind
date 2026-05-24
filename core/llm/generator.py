from core.prompts.prompt_registry import registry
from core.llm.manager import llm_manager
from core.llm.shared.types import Message, ProviderName, GenerationConfig

def _build_context(search_results: list, temp_file_content: str = None) -> str:
    """Build context string from optional file content and search results with labels."""
    context_parts = []
    
    if temp_file_content:
        context_parts.append(f"[SRC_1] (จาก: ไฟล์แนบของผู้ใช้)\n{temp_file_content}")
        
    if search_results:
        for i, (text, score) in enumerate(search_results):
            src_id = len(context_parts) + 1
            doc_name = "ไม่ระบุ"
            if text.startswith("[") and "]" in text:
                doc_name = text.split("]")[0].lstrip("[")
            context_parts.append(f"[SRC_{src_id}] (จาก: {doc_name})\n{text}")
            
    if not context_parts:
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"
        
    return "\n\n---\n\n".join(context_parts)

def _build_prompt(query: str, context: str) -> str:
    """Build the user prompt combining query and retrieved context."""
    return f"""ข้อมูลอ้างอิง:
{context}

---
คำถามของผู้ใช้: {query}

(ใช้ข้อมูลอ้างอิงข้างต้นในการตอบตามหลักการที่คุณได้รับมอบหมาย)
สำคัญมาก: หากคุณนำข้อมูลจาก [SRC_x] มาใช้ประกอบการตอบคำถาม ต้องใส่ตัวเลขอ้างอิง [x] ท้ายประโยคเหล่านั้นเสมอ (เช่น "ข้อมูลนี้กล่าวว่า... [1][2]") ห้ามสร้างตัวเลขอ้างอิงขึ้นมาเองเด็ดขาด"""

def generate(
    query: str, 
    search_results: list, 
    stream: bool = False, 
    provider: ProviderName = ProviderName.GEMINI,
    model_name: str = None,
    persona_id: str = "default",
    temp_file_content: str = None,
    chat_history: list = None
):
    """
    Generate an answer using the llm_manager.
    """
    # 1. Load system prompt from persona service or registry
    from services.persona_service import persona_service
    p_config = persona_service.get_persona(persona_id)
    persona_prompt = p_config.get("prompt", {}).get("system_role", "")
    
    # Base RAG Rules (Mandatory for all personas)
    base_prompt = """คุณคือผู้ช่วย AI ที่ทำงานร่วมกับระบบ RAG หน้าที่ของคุณคือตอบคำถามของผู้ใช้ โดยใช้ข้อมูลจากเอกสารที่แนบมาให้เป็นแกนหลักในการอ้างอิงข้อเท็จจริง.

แต่คุณมีอิสระอย่างเต็มที่ในการ:
1. สวมบทบาทและใช้น้ำเสียง ตามที่ระบุไว้ในบทบาทของคุณอย่างสุดพลัง
2. ใช้ความรู้รอบตัว มาช่วยอธิบาย ขยายความ หรือเปรียบเทียบให้เห็นภาพชัดเจนขึ้นได้ ตราบใดที่ไม่ขัดแย้งกับข้อเท็จจริงในเอกสาร

กฎเหล็กที่ต้องปฏิบัติตามอย่างเคร่งครัด:
1. ห้ามสร้างข้อมูลหรือข้อเท็จจริงขึ้นมาเอง (Hallucination) หากในเอกสารไม่มีข้อเท็จจริงเรื่องนั้นเลย และคุณจำเป็นต้องใช้ความรู้ตัวเองตอบ ให้เกริ่นบอกผู้ใช้นิดนึงว่า 'เรื่องนี้ไม่มีในเอกสารนะครับ แต่จากความรู้ทั่วไปของผม...'
2. ให้ใช้หางเสียง 'ครับ' ทุกครั้งที่จบประโยคหรือเมื่อเหมาะสม ห้ามใช้หางเสียง 'ค่ะ' หรือ 'ครับ/ค่ะ' โดยเด็ดขาด
"""
    
    if persona_prompt:
        system_prompt = f"{base_prompt}\n\nบทบาทและสไตล์การตอบของคุณ:\n{persona_prompt}"
    else:
        system_prompt = base_prompt
        
    # Append Tone if present
    tone = p_config.get("prompt", {}).get("tone", "")
    if tone and tone != "neutral":
        tone_map = {
            "polite": "สุภาพ",
            "formal": "เป็นทางการ",
            "friendly": "เป็นกันเอง",
            "concise": "กระชับ",
            "detailed": "ละเอียด",
            "humorous": "สนุกสนาน",
            "serious": "จริงจัง",
            "empathetic": "ให้กำลังใจ",
            "creative": "สร้างสรรค์"
        }
        active_tones = [tone_map.get(t.strip(), t.strip()) for t in tone.split(",") if t.strip() in tone_map]
        if active_tones:
            system_prompt += f"\n\nน้ำเสียงและสไตล์เพิ่มเติมที่ต้องใช้: {', '.join(active_tones)}"
        
    # Apply model config if present
    model_kwargs = p_config.get("model_config", {})
    
    # Filter valid keys for GenerationConfig
    import dataclasses
    valid_keys = {f.name for f in dataclasses.fields(GenerationConfig)}
    filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in valid_keys}
    config = GenerationConfig(**filtered_kwargs) if filtered_kwargs else None
    
    # 2. Build context and user prompt
    context = _build_context(search_results, temp_file_content)
    prompt = _build_prompt(query, context)
    
    # 3. Create Message objects
    messages = [
        Message(role="system", content=system_prompt)
    ]
    
    if chat_history:
        # Inject the last 5 turns (10 messages) of conversation history
        for msg in chat_history[-10:]:
            role = "assistant" if msg["role"] == "ai" else msg["role"]
            messages.append(Message(role=role, content=msg["content"]))
            
    messages.append(Message(role="user", content=prompt))
    
    # 4. Call llm_manager
    if stream:
        return llm_manager.generate_stream(provider, messages, model_name=model_name, config=config)
    else:
        return llm_manager.generate(provider, messages, model_name=model_name, config=config)
