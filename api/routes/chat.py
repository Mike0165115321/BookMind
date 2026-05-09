"""
Chat Routes — Endpoints for RAG Chat.
"""
from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse
from api.sse_handlers import classic_event_generator, agentic_event_generator

router = APIRouter(prefix="/api", tags=["Chat"])

@router.post("/ask")
async def ask_endpoint(request: Request):
    """
    Main RAG endpoint. SSE events: status, hyde, sources, token, done.
    """
    body = await request.json()
    query = body.get("query", "").strip()
    use_hyde = body.get("use_hyde", True)
    mode = body.get("mode", "classic")
    provider = body.get("provider", "gemini")
    model = body.get("model")

    if not query:
        return {"error": "กรุณาพิมพ์คำถาม"}

    if mode == "agentic":
        return EventSourceResponse(agentic_event_generator(query, use_hyde, provider, model))
    else:
        return EventSourceResponse(classic_event_generator(query, use_hyde, provider, model))

@router.get("/llm-models")
async def get_llm_models():
    """Get list of available models from all providers."""
    from core.llm.manager import llm_manager
    return llm_manager.get_all_available_models()
