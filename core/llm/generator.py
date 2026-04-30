"""
LLM Generator — High-level API for generating answers from retrieved context.

Coordinates building context/prompt and calling the LLM provider.
"""
from core.prompts.prompt_registry import registry
from core.llm.gemini_provider import GeminiProvider

# Default provider
default_provider = GeminiProvider()

def _build_context(search_results: list) -> str:
    """Build context string from search results."""
    if not search_results:
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"

    context_parts = []
    for i, (text, score) in enumerate(search_results, 1):
        context_parts.append(f"[แหล่งที่ {i}] (ความเกี่ยวข้อง: {score:.2f})\n{text}")

    return "\n\n---\n\n".join(context_parts)

def _build_prompt(query: str, context: str) -> str:
    """Build the user prompt combining query and retrieved context."""
    return f"""คำถาม: {query}

ข้อมูลอ้างอิง:
{context}

จากข้อมูลอ้างอิงข้างต้น:
- ตอบคำถามอย่างละเอียดและอ้างอิงแหล่งที่มา
- ถ้าเป็นคำถามเปรียบเทียบ: วิเคราะห์จุดเหมือน/ต่าง + สังเคราะห์เป็น framework ใหม่
- ถ้าเป็นคำถามเชิงกลยุทธ์: ให้ actionable steps + trade-offs + risk
- อย่าแค่สรุปแต่ละแหล่งแยกกัน → ต้องสังเคราะห์ข้ามแนวคิดให้เป็นคำตอบเดียวที่เชื่อมโยงกัน"""

def generate(query: str, search_results: list, stream: bool = False, provider=None):
    """
    Generate an answer using the provided LLM provider (defaults to Gemini).
    """
    provider = provider or default_provider
    
    # 1. Load system prompt from registry
    system_prompt = registry.get("rag_system")
    
    # 2. Build context and user prompt
    context = _build_context(search_results)
    prompt = _build_prompt(query, context)
    
    # 3. Call provider
    if stream:
        return provider.generate_stream(prompt, system_instruction=system_prompt)
    else:
        return provider.generate(prompt, system_instruction=system_prompt)
