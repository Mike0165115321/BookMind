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
    chat_id = body.get("chat_id") # Get chat_id from frontend

    if not query:
        return {"error": "กรุณาพิมพ์คำถาม"}

    if mode == "agentic":
        return EventSourceResponse(agentic_event_generator(query, use_hyde, provider, model, chat_id))
    else:
        return EventSourceResponse(classic_event_generator(query, use_hyde, provider, model, chat_id))

@router.get("/chats")
async def list_chats():
    from core.database import db
    return db.get_chats()

@router.get("/chats/{chat_id}/messages")
async def get_messages(chat_id: str):
    from core.database import db
    return db.get_messages(chat_id)

@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    from core.database import db
    db.delete_chat(chat_id)
    return {"status": "deleted"}

@router.get("/settings")
async def get_settings():
    from core.database import db
    return db.get_all_settings()

@router.post("/settings")
async def save_settings(request: Request):
    from core.database import db
    data = await request.json()
    for key, value in data.items():
        db.set_setting(key, value)
    return {"status": "saved"}

@router.get("/llm-models")
async def get_llm_models():
    """Get list of available models from all providers."""
    from core.llm.manager import llm_manager
    return llm_manager.get_all_available_models()
