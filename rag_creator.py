"""
RAG Creator — Builds FAISS index from source documents.
Reads .jsonl and .txt files, creates embeddings, and saves to disk.
"""
import os
import json
import torch
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import config


class RAGCreator:
    def __init__(self, model_name=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = model_name or config.MODEL_EMBEDDING
        print(f"📡 RAGCreator using device: {self.device.upper()}")
        self.model = SentenceTransformer(model_path, device=self.device)
        self.data = []

    def _load_jsonl(self, filepath):
        """Read a .jsonl file and extract combined text from each JSON line."""
        docs = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # Combine relevant fields into a single passage
                    parts = []
                    if obj.get("book_title"):
                        parts.append(f"[{obj['book_title']}]")
                    if obj.get("title"):
                        parts.append(obj["title"])
                    if obj.get("content"):
                        parts.append(obj["content"])
                    text = "\n".join(parts).strip()
                    if text:
                        docs.append(text)
                except json.JSONDecodeError:
                    continue
        return docs

    def build_and_save(self, source_path=None, save_dir=None, index_name=None):
        """Build FAISS index from source data and save to disk."""
        source_path = source_path or config.DATA_DIR
        save_dir = save_dir or config.STORAGE_DIR
        index_name = index_name or config.INDEX_NAME

        # 1. Create storage directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 2. Read data from source directory (supports .txt, .jsonl)
        print(f"📂 กำลังอ่านไฟล์จาก: {source_path}")
        for filename in sorted(os.listdir(source_path)):
            filepath = os.path.join(source_path, filename)
            if filename.endswith(".jsonl"):
                docs = self._load_jsonl(filepath)
                self.data.extend(docs)
            elif filename.endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        self.data.append(content)

        print(f"📊 โหลดเอกสารทั้งหมด {len(self.data)} chunks")

        # Guard: prevent crash if no data was loaded
        if not self.data:
            print("❌ ไม่พบข้อมูลที่จะสร้าง Index ได้ กรุณาตรวจสอบโฟลเดอร์ข้อมูล")
            return False

        # 3. Create Embeddings
        print("🧠 กำลังสร้าง Embeddings...")
        prefixed_docs = [f"passage: {doc}" for doc in self.data]
        embeddings = self.model.encode(
            prefixed_docs,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=config.BATCH_SIZE
        )
        embeddings_np = embeddings.cpu().detach().numpy().astype('float32')

        # 4. Create FAISS Index
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_np)
        print(f"🚀 FAISS Index created (Vectors: {index.ntotal}, Dimension: {dimension})")

        # 5. Save everything to disk
        faiss.write_index(index, os.path.join(save_dir, f"{index_name}.faiss"))
        with open(os.path.join(save_dir, f"{index_name}_data.pkl"), "wb") as f:
            pickle.dump(self.data, f)

        print(f"💾 บันทึกสำเร็จใน: {save_dir}")
        return True