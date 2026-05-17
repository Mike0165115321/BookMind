from core.prompts.prompt_registry import registry
from core.llm.manager import llm_manager
from core.llm.shared.types import Message, ProviderName, GenerationConfig

def _build_context(search_results: list) -> str:
    """Build context string from search results with document name labels."""
    if not search_results:
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"

    context_parts = []
    for i, (text, score) in enumerate(search_results):
        doc_name = "ไม่ระบุ"
        if text.startswith("[") and "]" in text:
            doc_name = text.split("]")[0].lstrip("[")
        
        context_parts.append(f"[SRC_{i+1}] (จาก: {doc_name})\n{text}")

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
    persona_id: str = "default"
):
    """
    Generate an answer using the llm_manager.
    """
    # 1. Load system prompt from persona service or registry
    from services.persona_service import persona_service
    p_config = persona_service.get_persona(persona_id)
    system_prompt = p_config.get("prompt", {}).get("system_role")
    
    if not system_prompt:
        system_prompt = registry.get("rag_system")
        
    # Apply model config if present
    model_kwargs = p_config.get("model_config", {})
    
    # 2. Build context and user prompt
    context = _build_context(search_results)
    prompt = _build_prompt(query, context)
    
    # 3. Create Message objects
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=prompt)
    ]
    
    # 4. Call llm_manager
    if stream:
        return llm_manager.generate_stream(provider, messages, model_name=model_name, **model_kwargs)
    else:
        return llm_manager.generate(provider, messages, model_name=model_name, **model_kwargs)
