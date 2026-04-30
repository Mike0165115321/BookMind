"""
Admin Routes — Endpoints for Document Management.
"""
import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from core.database import db
from core.ingestor import Ingestor
from services.chat_service import chat_service
import config

router = APIRouter(prefix="/api", tags=["Admin"])

# Initialize ingestor with the searcher from chat_service
ingestor = Ingestor(searcher_instance=chat_service.get_searcher())

@router.get("/documents")
async def get_documents():
    """List all documents and their ingestion status."""
    docs = db.get_all_documents()
    return {"documents": docs}

@router.post("/upload")
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
        os.makedirs(upload_dir, exist_ok=True)
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

@router.post("/retry/{doc_id}")
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

@router.delete("/documents/{doc_id}")
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
