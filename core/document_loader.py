import os
import json
import fitz  # PyMuPDF
import docx  # python-docx

class DocumentLoader:
    """
    Dedicated class to handle text extraction from various file formats.
    """
    @staticmethod
    def load(filepath, book_title=None):
        filename = os.path.basename(filepath).lower()
        
        if filename.endswith((".jsonl", ".json")):
            return DocumentLoader._load_json(filepath, book_title)
        elif filename.endswith(".pdf"):
            return DocumentLoader._load_pdf(filepath, book_title)
        elif filename.endswith((".docx", ".doc")):
            return DocumentLoader._load_docx(filepath, book_title)
        else:
            return DocumentLoader._load_text(filepath, book_title)

    @staticmethod
    def _load_json(filepath, book_title=None):
        docs = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content.startswith('[') and content.endswith(']'):
                    data = json.loads(content)
                    for obj in data:
                        docs.append(DocumentLoader._format_json_obj(obj, book_title))
                    return docs
        except:
            pass

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    docs.append(DocumentLoader._format_json_obj(obj, book_title))
                except: continue
        return docs

    @staticmethod
    def _format_json_obj(obj, book_title=None):
        b_title = book_title or obj.get("book_title", "")
        title = obj.get("title", "")
        content = obj.get("content", "")
        
        prefix = ""
        if b_title: prefix += f"[{b_title}] "
        if title: prefix += f"{title}\n"
        
        return {"content": content, "metadata_prefix": prefix.strip()}

    @staticmethod
    def _load_pdf(filepath, book_title=None):
        text = ""
        try:
            with fitz.open(filepath) as doc:
                for page in doc:
                    text += page.get_text() + "\n"
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
        return [{"content": text, "metadata_prefix": f"[{book_title}]" if book_title else ""}]

    @staticmethod
    def _load_docx(filepath, book_title=None):
        text = ""
        try:
            doc = docx.Document(filepath)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            print(f"❌ Error loading DOCX: {e}")
        return [{"content": text, "metadata_prefix": f"[{book_title}]" if book_title else ""}]

    @staticmethod
    def _load_text(filepath, book_title=None):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return [{"content": f.read(), "metadata_prefix": f"[{book_title}]" if book_title else ""}]
        except:
            return []
