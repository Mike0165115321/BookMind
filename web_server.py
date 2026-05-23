"""
Web Server — FastAPI backend for RAG Web UI.

Modular entry point that mounts routers and serves static files.
"""
import uvicorn
# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]
from fastapi.responses import FileResponse
import os

import config
from api.routes import chat, admin

from contextlib import asynccontextmanager

# ──────────────────────────────────────────────
# Lifespan Events
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure upload directory exists at startup
    os.makedirs(os.path.join(config.DATA_DIR, "uploads"), exist_ok=True)
    print("🚀 BookMind Web Server ready! (Modular Mode)")
    yield

# ──────────────────────────────────────────────
# Initialize
# ──────────────────────────────────────────────
app = FastAPI(
    title="BookMind RAG Knowledge Base", 
    version="3.0",
    description="Refactored Modular RAG System",
    lifespan=lifespan
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
async def get_index():
    return FileResponse("web/index.html")

@app.get("/settings")
async def get_settings_page():
    return FileResponse("web/settings.html")

@app.get("/admin")
async def get_admin_page():
    return FileResponse("web/admin.html")

# Mount static directory for JS/CSS
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("web_server:app", host=config.APP_HOST, port=config.APP_PORT, reload=False)

