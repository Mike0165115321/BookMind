"""
Web Server — FastAPI backend for RAG Web UI.

Modular entry point that mounts routers and serves static files.
"""
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

import config
from api.routes import chat, admin

# ──────────────────────────────────────────────
# Initialize
# ──────────────────────────────────────────────
app = FastAPI(
    title="BookMind RAG Knowledge Base", 
    version="3.0",
    description="Refactored Modular RAG System"
)

# ──────────────────────────────────────────────
# Include Routers
# ──────────────────────────────────────────────
app.include_router(chat.router)
app.include_router(admin.router)

# ──────────────────────────────────────────────
# Static Files & Frontend
# ──────────────────────────────────────────────
@app.get("/")
async def serve_index():
    return FileResponse("web/index.html")

@app.get("/admin")
async def serve_admin():
    return FileResponse("web/admin.html")

# Mount static directory for JS/CSS
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Ensure upload directory exists at startup
@app.on_event("startup")
async def startup():
    os.makedirs(os.path.join(config.DATA_DIR, "uploads"), exist_ok=True)
    print("🚀 BookMind Web Server ready! (Modular Mode)")

# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("web_server:app", host=config.APP_HOST, port=config.APP_PORT, reload=False)
