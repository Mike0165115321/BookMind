"""
Web Server — FastAPI backend for RAG Web UI.

Serves the frontend and provides a streaming API endpoint.
SSE (Server-Sent Events) is used for real-time token streaming.

Usage:
  python3 web_server.py
  → Open http://localhost:8000
"""
import json
import time
import asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

import config
from rag_searcher import RAGSearcher
from core.llm_generator import generate
from core.query_transformer import hyde_transform

# ──────────────────────────────────────────────
# Initialize
# ──────────────────────────────────────────────
app = FastAPI(title="RAG Knowledge Base", version="1.0")

# Load RAG index once at startup
searcher = None

@app.on_event("startup")
async def startup():
    global searcher
    searcher = RAGSearcher()
    searcher.load_index()
    print("🚀 RAG Web Server ready!")


# ──────────────────────────────────────────────
# API Endpoint — SSE Streaming
# ──────────────────────────────────────────────
@app.post("/api/ask")
async def ask_endpoint(request: Request):
    """
    Main RAG endpoint. Accepts JSON { query, use_hyde }.
    Returns SSE stream with events: search, sources, token, done.
    """
    body = await request.json()
    query = body.get("query", "").strip()
    use_hyde = body.get("use_hyde", True)

    if not query:
        return {"error": "กรุณาพิมพ์คำถาม"}

    async def event_generator():
        t_total = time.time()

        # Stage 0: HyDE Query Transform
        search_query = query
        hyde_time = 0
        if use_hyde and config.ENABLE_HYDE:
            yield {"event": "status", "data": json.dumps({"stage": "hyde", "message": "🪄 กำลังสร้าง HyDE..."})}
            t_hyde = time.time()
            search_query = await asyncio.to_thread(hyde_transform, query)
            hyde_time = time.time() - t_hyde
            yield {"event": "hyde", "data": json.dumps({"hyde_query": search_query[:200], "time": round(hyde_time, 2)})}

        # Stage 1+2: Hybrid Search + Adaptive Reranking
        yield {"event": "status", "data": json.dumps({"stage": "search", "message": "🔍 กำลังค้นหา..."})}
        t_search = time.time()
        results = await asyncio.to_thread(searcher.search, search_query, config.TOP_K_RETRIEVAL)
        search_time = time.time() - t_search

        # Send sources
        sources = []
        for i, (text, score) in enumerate(results[:config.TOP_K_DISPLAY]):
            # Extract book title
            title = text.split("]")[0].lstrip("[") if "[" in text else "ไม่ระบุ"
            sources.append({
                "rank": i + 1,
                "title": title,
                "text": text[:300],
                "score": round(float(score), 3),
            })
        yield {"event": "sources", "data": json.dumps({"sources": sources, "search_time": round(search_time, 3)}, ensure_ascii=False)}

        # Stage 3: LLM Generation (streaming)
        yield {"event": "status", "data": json.dumps({"stage": "generate", "message": f"🤖 กำลังสร้างคำตอบ ({config.GEMINI_MODEL})..."})}
        t_gen = time.time()

        for chunk in generate(query, results[:config.TOP_K_DISPLAY], stream=True):
            yield {"event": "token", "data": json.dumps({"text": chunk})}
            await asyncio.sleep(0)  # yield control for responsiveness

        gen_time = time.time() - t_gen
        total_time = time.time() - t_total

        # Done event with timing
        yield {"event": "done", "data": json.dumps({
            "hyde_time": round(hyde_time, 2),
            "search_time": round(search_time, 3),
            "gen_time": round(gen_time, 2),
            "total_time": round(total_time, 2),
        })}

    return EventSourceResponse(event_generator())


# ──────────────────────────────────────────────
# Serve Frontend
# ──────────────────────────────────────────────
@app.get("/")
async def serve_index():
    return FileResponse("web/index.html")

app.mount("/static", StaticFiles(directory="web"), name="static")


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("web_server:app", host="0.0.0.0", port=8000, reload=False)
