import os
import asyncio
from core.database import db
from rag_creator import RAGCreator
import config

class Ingestor:
    def __init__(self, searcher_instance=None):
        self.creator = RAGCreator()
        self.searcher = searcher_instance

    async def process_document(self, doc_id, filepath, book_title=None, category=None):
        """
        Background task to process a document and update the index.
        """
        try:
            # 1. Update status to PROCESSING
            db.update_status(doc_id, "PROCESSING")
            print(f"🏗️ [DEBUG-V3] เริ่มประมวลผลเอกสาร ID: {doc_id} ({filepath})")

            # 2. Extract and chunk
            # Run blocking CPU/GPU code in a thread to not block the event loop
            chunks = await asyncio.to_thread(
                self.creator.process_single_file, 
                filepath, 
                book_title, 
                category
            )
            
            if not chunks:
                raise ValueError("ไม่สามารถแยกข้อมูลจากไฟล์ได้ หรือไฟล์ว่างเปล่า")

            # 3. Update index on disk
            success = await asyncio.to_thread(
                self.creator.update_index,
                chunks
            )
            
            if not success:
                raise RuntimeError("ไม่สามารถอัปเดต Index ได้")

            # 4. Success! Update DB
            db.update_status(doc_id, "COMPLETED", total_chunks=len(chunks))
            print(f"✅ ประมวลผลเอกสาร {doc_id} สำเร็จ! ({len(chunks)} chunks)")

            # 5. Reload searcher index if available
            if self.searcher:
                await asyncio.to_thread(self.searcher.reload_index)

        except Exception as e:
            error_msg = str(e)
            print(f"❌ เกิดข้อผิดพลาดในการประมวลผล {doc_id}: {error_msg}")
            db.update_status(doc_id, "ERROR", error_message=error_msg)

# We'll initialize this in web_server.py to pass the running searcher instance
