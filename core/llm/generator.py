from core.prompts.prompt_registry import registry
from core.llm.manager import llm_manager
from core.llm.shared.types import Message, ProviderName, GenerationConfig

def _build_context(search_results: list) -> str:
    """Build context string from search results with document name labels."""
    if not search_results:
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"

    context_parts = []
    for text, score in search_results:
        doc_name = "ไม่ระบุ"
        if text.startswith("[") and "]" in text:
            doc_name = text.split("]")[0].lstrip("[")
        
        context_parts.append(f"(จาก: {doc_name})\n{text}")

    return "\n\n---\n\n".join(context_parts)

def _build_prompt(query: str, context: str) -> str:
    """Build the user prompt combining query and retrieved context."""
    return f"""ข้อมูลอ้างอิง:
{context}

---
คำถามของผู้ใช้: {query}

(ใช้ข้อมูลอ้างอิงข้างต้นในการตอบตามหลักการที่คุณได้รับมอบหมาย)"""

def generate(
    query: str, 
    search_results: list, 
    stream: bool = False, 
    provider: ProviderName = ProviderName.GEMINI,
    model_name: str = None
):
    """
    Generate an answer using the llm_manager.
    """
    # 1. Load system prompt from registry
    system_prompt = registry.get("rag_system")
    
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
        return llm_manager.generate_stream(provider, messages, model_name=model_name)
    else:
        return llm_manager.generate(provider, messages, model_name=model_name)
