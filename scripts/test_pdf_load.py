
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.document_loader import DocumentLoader
import config

def test_load():
    filepath = "/home/mikedev/BookMind/data/uploads/Aetox_Foundation_Document_v1.pdf"
    print(f"📄 Testing load for: {filepath}")
    
    if not os.path.exists(filepath):
        print("❌ File not found!")
        return

    docs = DocumentLoader.load(filepath)
    if not docs:
        print("❌ No content loaded!")
        return
    
    content = docs[0]["content"]
    print(f"📊 Total characters extracted: {len(content)}")
    print(f"📝 Preview (first 500 chars):\n{content[:500]}")
    print(f"📝 Preview (last 500 chars):\n{content[-500:]}")

if __name__ == "__main__":
    test_load()
