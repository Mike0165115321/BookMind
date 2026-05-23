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

