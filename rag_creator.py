"""
RAG Creator — Builds FAISS index + BM25 corpus from source documents.
Reads .jsonl and .txt files, chunks them intelligently, creates embeddings,
and saves both FAISS index and tokenized BM25 data to disk.
"""
import os
import re
import json
import torch
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import config


class TextChunker:
    """Splits long text into overlapping chunks at natural boundaries."""

    def __init__(self, chunk_size=None, chunk_overlap=None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    def chunk(self, text, metadata_prefix=""):
        """
        Split text into chunks with overlap.
        Each chunk gets the metadata_prefix prepended (e.g. [Book Title]).

        Strategy:
          1. Split by paragraphs first
          2. Merge small paragraphs until near chunk_size
          3. If a single paragraph exceeds chunk_size, split at sentence boundaries
          4. Add overlap from previous chunk's tail
        """
        if not text.strip():
            return []

        full_text = f"{metadata_prefix}\n{text}".strip() if metadata_prefix else text.strip()
        if len(full_text) <= self.chunk_size:
            return [full_text]

        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 1 > self.chunk_size:
                if current_chunk:
                    chunk_text = f"{metadata_prefix}\n{current_chunk}".strip() if metadata_prefix else current_chunk
                    chunks.append(chunk_text)
                    overlap = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk

                    if len(para) > self.chunk_size:
                        sub_chunks = self._split_long_paragraph(para, metadata_prefix)
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = overlap + "\n" + para
                else:
                    if len(para) > self.chunk_size:
                        sub_chunks = self._split_long_paragraph(para, metadata_prefix)
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = para
            else:
                current_chunk = f"{current_chunk}\n{para}".strip() if current_chunk else para

        if current_chunk:
            chunk_text = f"{metadata_prefix}\n{current_chunk}".strip() if metadata_prefix else current_chunk
            chunks.append(chunk_text)

        return chunks

    def _split_long_paragraph(self, text, metadata_prefix=""):
        """Split a single long paragraph at sentence boundaries."""
        sentences = re.split(r'(?<=[.。!?\n])\s*', text)
        chunks = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 > self.chunk_size:
                if current:
                    chunk_text = f"{metadata_prefix}\n{current}".strip() if metadata_prefix else current
                    chunks.append(chunk_text)
                    overlap = current[-self.chunk_overlap:] if len(current) > self.chunk_overlap else current
                    current = overlap + " " + sentence
                else:
                    chunk_text = f"{metadata_prefix}\n{sentence[:self.chunk_size]}".strip() if metadata_prefix else sentence[:self.chunk_size]
                    chunks.append(chunk_text)
                    current = sentence[self.chunk_size - self.chunk_overlap:]
            else:
                current = f"{current} {sentence}".strip() if current else sentence

        if current:
            chunk_text = f"{metadata_prefix}\n{current}".strip() if metadata_prefix else current
            chunks.append(chunk_text)

        return chunks


def tokenize_thai(text):
    """
    Simple word-level tokenizer for Thai + English mixed text.
    Splits on whitespace, punctuation, and Thai character boundaries.
    Good enough for BM25 — no external dependency needed.
    """
    # Split on whitespace and common punctuation, keep words
    tokens = re.findall(r'[\u0E00-\u0E7F]+|[a-zA-Z0-9]+', text.lower())
    # Filter very short tokens (single chars in Thai are usually not meaningful for BM25)
    return [t for t in tokens if len(t) > 1]


class RAGCreator:
    def __init__(self, model_name=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = model_name or config.MODEL_EMBEDDING
        print(f"📡 RAGCreator using device: {self.device.upper()}")
        self.model = SentenceTransformer(model_path, device=self.device)
        self.data = []
        self.chunker = TextChunker()

    def _load_jsonl(self, filepath):
        """Read a .jsonl file and extract chunked text from each JSON line."""
        docs = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    prefix_parts = []
                    if obj.get("book_title"):
                        prefix_parts.append(f"[{obj['book_title']}]")
                    if obj.get("title"):
                        prefix_parts.append(obj["title"])
                    metadata_prefix = "\n".join(prefix_parts)

                    content = obj.get("content", "")
                    if not content:
                        full_text = metadata_prefix
                        if full_text.strip():
                            docs.append(full_text.strip())
                        continue

                    chunks = self.chunker.chunk(content, metadata_prefix=metadata_prefix)
                    docs.extend(chunks)
                except json.JSONDecodeError:
                    continue
        return docs

    def build_and_save(self, source_path=None, save_dir=None, index_name=None):
        """Build FAISS index + BM25 tokenized corpus and save to disk."""
        source_path = source_path or config.DATA_DIR
        save_dir = save_dir or config.STORAGE_DIR
        index_name = index_name or config.INDEX_NAME

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 1. Read & chunk data
        print(f"📂 กำลังอ่านไฟล์จาก: {source_path}")
        print(f"✂️  Chunking: {config.CHUNK_SIZE} chars, overlap {config.CHUNK_OVERLAP} chars")
        for filename in sorted(os.listdir(source_path)):
            filepath = os.path.join(source_path, filename)
            if filename.endswith(".jsonl"):
                docs = self._load_jsonl(filepath)
                self.data.extend(docs)
            elif filename.endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        chunks = self.chunker.chunk(content)
                        self.data.extend(chunks)

        print(f"📊 โหลดเอกสารทั้งหมด {len(self.data)} chunks")

        if not self.data:
            print("❌ ไม่พบข้อมูลที่จะสร้าง Index ได้ กรุณาตรวจสอบโฟลเดอร์ข้อมูล")
            return False

        # 2. Create Embeddings (for Dense Search)
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

        # 3. Create FAISS Index
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_np)
        print(f"🚀 FAISS Index created (Vectors: {index.ntotal}, Dimension: {dimension})")

        # 4. Tokenize for BM25 (keyword search)
        print("📝 กำลังสร้าง BM25 corpus...")
        tokenized_corpus = [tokenize_thai(doc) for doc in self.data]
        avg_tokens = sum(len(t) for t in tokenized_corpus) / len(tokenized_corpus)
        print(f"📝 BM25 corpus ready (avg {avg_tokens:.0f} tokens/chunk)")

        # 5. Save everything to disk
        faiss.write_index(index, os.path.join(save_dir, f"{index_name}.faiss"))
        with open(os.path.join(save_dir, f"{index_name}_data.pkl"), "wb") as f:
            pickle.dump(self.data, f)
        with open(os.path.join(save_dir, f"{index_name}_bm25.pkl"), "wb") as f:
            pickle.dump(tokenized_corpus, f)

        print(f"💾 บันทึกสำเร็จใน: {save_dir}")
        return True