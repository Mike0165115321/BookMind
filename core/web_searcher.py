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
            with DDGS() as ddgs:
                # Use text search
                ddgs_results = list(ddgs.text(query, max_results=max_results))
                for r in ddgs_results:
                    title = r.get("title", "")
                    href = r.get("href", "")
                    body = r.get("body", "")
                    # Format exactly like our RAG chunks for compatibility
                    formatted_text = f"[Web: {title}] {body}\n(Source: {href})"
                    # Give it a high score so it shows up at the top
                    results.append((formatted_text, 0.95))
        except Exception as e:
            print(f"   ❌ Web Search failed: {e}")
            
        return results
