"""
Chat Routes — Endpoints for RAG Chat.
"""
from fastapi import APIRouter, Request, File, UploadFile
from sse_starlette.sse import EventSourceResponse
from api.sse_handlers import classic_event_generator, agentic_event_generator
import os
import uuid
import config

router = APIRouter(prefix="/api", tags=["Chat"])

@router.post("/chat/upload_temp")
async def upload_temp_file(file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{ext}"
        upload_dir = os.path.join(config.DATA_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, unique_filename)
        
        with open(file_path, "wb") as f:
            f.write(await file.read())
            
        return {"status": "uploaded", "file_path": file_path, "original_name": file.filename}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.post("/ask")
async def ask_endpoint(request: Request):
    """
    Main RAG endpoint. SSE events: status, hyde, sources, token, done.
    """
    body = await request.json()
    query = body.get("query", "").strip()
    use_hyde = body.get("use_hyde", True)
    mode = body.get("mode", "classic")
    provider = body.get("provider")
    model = body.get("model")
    chat_id = body.get("chat_id") # Get chat_id from frontend
    persona_id = body.get("persona_id")
    temp_file_path = body.get("temp_file_path")
    temp_file_name = body.get("temp_file_name")
    
    from core.database import db
    if not persona_id:
        persona_id = db.get_setting("persona_id", "default")
    if not provider:
        provider = db.get_setting("gen_provider", "gemini")
    if not model:
        model = db.get_setting("gen_model")

    if not query:
        return {"error": "กรุณาพิมพ์คำถาม"}

    if mode == "agentic":
        return EventSourceResponse(agentic_event_generator(query, use_hyde, provider, model, chat_id, persona_id, temp_file_path, temp_file_name))
    else:
        return EventSourceResponse(classic_event_generator(query, use_hyde, provider, model, chat_id, persona_id, temp_file_path, temp_file_name))

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

@router.get("/personas")
async def get_personas():
    """Get list of available personas."""
    from services.persona_service import persona_service
    return persona_service.get_all_personas()

@router.post("/personas")
async def create_persona(request: Request):
    """Create a new custom persona."""
    from services.persona_service import persona_service
    data = await request.json()
    
    label = data.get("label", "Custom Persona")
    description = data.get("description", "Custom built persona")
    system_role = data.get("system_role", "")
    tone = data.get("tone", "neutral")
    language = data.get("language", "th")
    temperature = data.get("temperature", 0.5)
    
    if not system_role:
        return {"error": "system_role is required"}
        
    persona_id = persona_service.add_persona(label, description, system_role, tone, language, float(temperature))
    return {"status": "created", "id": persona_id}
