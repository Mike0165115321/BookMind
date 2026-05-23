"""
Web Searcher using DuckDuckGo
"""
# pyrefly: ignore [missing-import]
from duckduckgo_search import DDGS
import time

class WebSearcher:
    @staticmethod
    def search(query: str, max_results: int = 3) -> list:
        print(f"   🌐 Web Searching: {query}")
        results = []
        
        try:
            # pyrefly: ignore [missing-import]
            from duckduckgo_search import DDGS
        except ImportError as e:
            print(f"   ❌ duckduckgo_search import failed: {e}")
            return results

        # Rule-based simple Thai greeting & filler purification
        clean_query = query.strip()
        
        thai_fillers = [
            "สวัสดีครับ", "สวัสดีค่ะ", "สวัสดี", "ดีครับ", "ดีค่ะ", 
            "ช่วยหาหน่อย", "ช่วยค้นหาหน่อย", "ช่วยหา", "ช่วยค้นหา",
            "ตอนนี้มีอะไรบ้าง", "มีอะไรบ้าง", "หน่อยครับ", "หน่อยค่ะ", "หน่อย"
        ]
        
        for filler in sorted(thai_fillers, key=len, reverse=True):
            clean_query = clean_query.replace(filler, "")
            
        # Clean up punctuation marks and special symbols that degrade search engine accuracy
        punctuation_to_remove = ['"', "'", '`', '(', ')', '[', ']', '{', '}', ',', '.', '!', '?', '-', '_', '+', '=', '*', '/', '\\', '|', '&', '^', '%', '$', '#', '@']
        for p in punctuation_to_remove:
            clean_query = clean_query.replace(p, " ")
            
        # Standardize whitespace spacing
        clean_query = " ".join(clean_query.split()).strip()
        
        # Word-level truncation to prevent query bloat
        words = clean_query.split()
        if len(words) > 8:
            clean_query = " ".join(words[:8])
            
        # Character-level truncation safeguard
        if len(clean_query) > 100:
            clean_query = clean_query[:100].strip()
            
        if not clean_query:
            clean_query = query.strip()
        
        # Try multiple backends sequentially to ensure high resilience against rate limits & blocks
        for backend in ["lite", "html", "api"]:
            try:
                print(f"   🌐 Trying DuckDuckGo text search (backend='{backend}')...")
                with DDGS() as ddgs:
                    try:
                        ddgs_results = list(ddgs.text(clean_query, backend=backend, max_results=max_results))
                    except TypeError:
                        # Fallback for older versions that don't support the 'backend' keyword argument
                        print(f"   ⚠️ 'backend' argument not supported. Falling back to standard search.")
                        ddgs_results = list(ddgs.text(clean_query, max_results=max_results))

                    if ddgs_results:
                        for r in ddgs_results:
                            title = r.get("title", "")
                            href = r.get("href", "")
                            body = r.get("body", "")
                            # Format exactly like RAG chunks for compatibility
                            formatted_text = f"[Web: {title}] {body}\n(Source: {href})"
                            results.append((formatted_text, 0.95))
                        print(f"   ✅ Web Search succeeded using backend='{backend}'! Found {len(results)} results.")
                        break
            except Exception as e:
                print(f"   ⚠️ DDGS search failed with backend='{backend}': {e}")
                
        return results

