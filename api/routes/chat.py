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

    if not query:
        return {"error": "กรุณาพิมพ์คำถาม"}

    if mode == "agentic":
        return EventSourceResponse(agentic_event_generator(query, use_hyde))
    else:
        return EventSourceResponse(classic_event_generator(query, use_hyde))

@router.post("/search")
async def search_endpoint(request: Request):
    """
    Direct Retrieval endpoint for n8n tools. Returns raw context chunks.
    """
    from services.chat_service import pipeline
    
    body = await request.json()
    query = body.get("query", "").strip()
    top_k = body.get("top_k", 5)

    if not query:
        return {"results": []}

    # ค้นหาด้วย RAG Pipeline เดิมของคุณ (Dense + BM25 + Rerank)
    results = pipeline.search(query, top_k=top_k)
    
    # แปลงผลลัพธ์เป็น JSON สำหรับ n8n
    search_results = []
    for doc, score in results:
        search_results.append({
            "content": doc.get("content", ""),
            "metadata": {
                "book_title": doc.get("book_title", ""),
                "title": doc.get("title", ""),
                "score": float(score)
            }
        })
    
    return {"results": search_results}
