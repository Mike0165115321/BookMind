"""
Web Server — FastAPI backend for RAG Web UI.

Serves the frontend and provides streaming API endpoints.
SSE (Server-Sent Events) is used for real-time token streaming.

Supports two modes:
  - Classic:  /api/ask with mode="classic" (default)
  - Agentic:  /api/ask with mode="agentic" (multi-hop)

Usage:
  python3 web_server.py
  → Open http://localhost:8000
"""
import json
import time
import asyncio
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
import os
import shutil

import config
from rag_searcher import RAGSearcher
from core.llm_generator import generate
from core.query_transformer import hyde_transform
from core.agentic_controller import AgenticController
from core.database import db
from core.ingestor import Ingestor

# ──────────────────────────────────────────────
# Initialize
# ──────────────────────────────────────────────
app = FastAPI(title="RAG Knowledge Base", version="2.0")

# Global instances
searcher = None
ingestor = None

@app.on_event("startup")
async def startup():
    global searcher, ingestor
    searcher = RAGSearcher()
    searcher.load_index()
    
    # Initialize ingestor with the running searcher
    ingestor = Ingestor(searcher_instance=searcher)
    
    # Ensure upload directory exists
    os.makedirs(os.path.join(config.DATA_DIR, "uploads"), exist_ok=True)
    
    print("🚀 RAG Web Server ready! (Classic + Agentic modes)")


# ──────────────────────────────────────────────
# API Endpoint — SSE Streaming (Classic + Agentic)
# ──────────────────────────────────────────────
@app.post("/api/ask")
async def ask_endpoint(request: Request):
    """
    Main RAG endpoint. Accepts JSON { query, use_hyde, mode }.

    mode="classic" (default):
      SSE events: status, hyde, sources, token, done

    mode="agentic":
      SSE events: status, decompose, search_iteration, evaluate,
                  sources, token, done
    """
    body = await request.json()
    query = body.get("query", "").strip()
    use_hyde = body.get("use_hyde", True)
    mode = body.get("mode", "classic")

    if not query:
        return {"error": "กรุณาพิมพ์คำถาม"}

    if mode == "agentic":
        return EventSourceResponse(_agentic_event_generator(query, use_hyde))
    else:
        return EventSourceResponse(_classic_event_generator(query, use_hyde))


# ──────────────────────────────────────────────
# Admin API — Ingestion Management
# ──────────────────────────────────────────────
@app.get("/api/documents")
async def get_documents():
    """List all documents and their ingestion status."""
    docs = db.get_all_documents()
    return {"documents": docs}


@app.post("/api/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    book_title: str = Form(None),
    category: str = Form(None)
):
    """Upload a document file and register it in the database."""
    try:
        # Save file to uploads folder
        upload_dir = os.path.join(config.DATA_DIR, "uploads")
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Register in DB
        doc_id = db.add_document(
            filename=file.filename,
            book_title=book_title,
            category=category
        )
        
        # Trigger ingestion in background
        background_tasks.add_task(
            ingestor.process_document, 
            doc_id, 
            file_path, 
            book_title, 
            category
        )
        
        return {
            "success": True, 
            "message": f"อัปโหลดไฟล์ {file.filename} เรียบร้อย ระบบกำลังประมวลผลเบื้องหลัง",
            "doc_id": doc_id
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error: {str(e)}"}
        )


@app.post("/api/retry/{doc_id}")
async def retry_upload(doc_id: int, background_tasks: BackgroundTasks):
    doc = db.get_document(doc_id)
    if not doc:
        return {"success": False, "message": "ไม่พบเอกสาร"}
    
    # Reconstruct path from filename
    upload_dir = os.path.join(config.DATA_DIR, "uploads")
    file_path = os.path.join(upload_dir, doc['filename'])
    
    # Check if file still exists
    if not os.path.exists(file_path):
        return {"success": False, "message": "ไม่พบไฟล์ต้นฉบับบน Server"}
        
    background_tasks.add_task(
        ingestor.process_document,
        doc_id,
        file_path,
        doc['book_title'],
        doc['category']
    )
    return {"success": True, "message": "กำลังเริ่มประมวลผลใหม่"}

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int):
    """Delete a document entry from the database."""
    try:
        db.delete_document(doc_id)
        return {"success": True, "message": "ลบข้อมูลสำเร็จ"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error: {str(e)}"}
        )


# ──────────────────────────────────────────────
# Classic Pipeline (SSE Generator)
# ──────────────────────────────────────────────
async def _classic_event_generator(query: str, use_hyde: bool):
    """Original single-shot pipeline with SSE streaming."""
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
        await asyncio.sleep(0)

    gen_time = time.time() - t_gen
    total_time = time.time() - t_total

    yield {"event": "done", "data": json.dumps({
        "mode": "classic",
        "hyde_time": round(hyde_time, 2),
        "search_time": round(search_time, 3),
        "gen_time": round(gen_time, 2),
        "total_time": round(total_time, 2),
    })}


# ──────────────────────────────────────────────
# Agentic Pipeline (SSE Generator)
# ──────────────────────────────────────────────
async def _agentic_event_generator(query: str, use_hyde: bool):
    """Multi-hop agentic pipeline with SSE streaming."""
    t_total = time.time()

    controller = AgenticController(searcher=searcher, use_hyde=use_hyde)

    # Run the agentic pipeline in a thread (it's blocking internally)
    # We use run_stream_with_answer to get events
    yield {
        "event": "status",
        "data": json.dumps({
            "stage": "decompose",
            "message": "🧠 กำลังวิเคราะห์คำถาม...",
        }),
    }

    # Run agentic pipeline synchronously in a thread, collecting events
    events = await asyncio.to_thread(
        lambda: list(controller.run_stream_with_answer(query))
    )

    # Replay events as SSE
    for event in events:
        if event.event_type == "decompose":
            d = event.data
            msg = f"🔀 แยกเป็น {len(d['sub_queries'])} sub-queries"
            yield {
                "event": "decompose",
                "data": json.dumps({
                    "query_type": d["query_type"],
                    "sub_queries": d["sub_queries"],
                    "reasoning": d["reasoning"],
                    "message": msg,
                }, ensure_ascii=False),
            }

        elif event.event_type == "search_start":
            d = event.data
            msg = f"🔍 ค้นหารอบ {d['iteration']}/{d['total_iterations']}: {d['query'][:80]}"
            yield {
                "event": "search_iteration",
                "data": json.dumps({
                    "iteration": d["iteration"],
                    "query": d["query"],
                    "message": msg,
                }, ensure_ascii=False),
            }
            yield {
                "event": "status",
                "data": json.dumps({
                    "stage": "search",
                    "message": msg,
                }, ensure_ascii=False),
            }

        elif event.event_type == "search_done":
            d = event.data
            yield {
                "event": "search_done",
                "data": json.dumps({
                    "iteration": d["iteration"],
                    "query": d["query"],
                    "num_results": d["num_results"],
                    "new_chunks": d["new_chunks"],
                    "total_chunks": d["total_chunks"],
                }, ensure_ascii=False),
            }

        elif event.event_type == "evaluate":
            d = event.data
            status = "✅ ข้อมูลเพียงพอ" if d["is_sufficient"] else f"🔄 ยังไม่ครบ (confidence={d['confidence']:.0%})"
            yield {
                "event": "evaluate",
                "data": json.dumps({
                    "is_sufficient": d["is_sufficient"],
                    "confidence": d["confidence"],
                    "missing_aspects": d["missing_aspects"],
                    "message": status,
                }, ensure_ascii=False),
            }
            yield {
                "event": "status",
                "data": json.dumps({
                    "stage": "evaluate",
                    "message": status,
                }, ensure_ascii=False),
            }

        elif event.event_type == "sources":
            d = event.data
            yield {
                "event": "sources",
                "data": json.dumps({
                    "sources": d["sources"],
                    "search_time": 0,
                    "iterations": d.get("iterations", 1),
                    "total_chunks": d.get("total_chunks", 0),
                }, ensure_ascii=False),
            }

        elif event.event_type == "synthesize":
            d = event.data
            yield {
                "event": "status",
                "data": json.dumps({
                    "stage": "generate",
                    "message": f"🤖 สังเคราะห์คำตอบจาก {d['total_chunks']} chunks ({d['iterations']} iterations)...",
                }),
            }

        elif event.event_type == "token":
            yield {
                "event": "token",
                "data": json.dumps({"text": event.data["text"]}),
            }

        elif event.event_type == "done":
            d = event.data
            total_time = time.time() - t_total
            yield {
                "event": "done",
                "data": json.dumps({
                    "mode": "agentic",
                    "total_time": round(total_time, 2),
                    "iterations": d.get("iterations", 1),
                    "query_type": d.get("query_type", "simple"),
                    "sub_queries": d.get("sub_queries", []),
                    "total_chunks": d.get("total_chunks", 0),
                }),
            }

        await asyncio.sleep(0)  # Yield control


# ──────────────────────────────────────────────
# Serve Frontend
# ──────────────────────────────────────────────
@app.get("/")
async def serve_index():
    return FileResponse("web/index.html")


@app.get("/admin")
async def serve_admin():
    return FileResponse("web/admin.html")

app.mount("/static", StaticFiles(directory="web"), name="static")


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("web_server:app", host="0.0.0.0", port=8000, reload=False)
