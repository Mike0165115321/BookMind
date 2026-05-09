import http.client
import json
from core.llm.ollama.config import ollama_config

def check_ollama_health() -> bool:
    """Check if Ollama service is running."""
    try:
        from urllib.parse import urlparse
        url = urlparse(ollama_config.base_url)
        conn = http.client.HTTPConnection(url.hostname, url.port, timeout=2)
        conn.request("GET", "/api/tags")
        res = conn.getresponse()
        return res.status == 200
    except:
        return False

def map_ollama_error(error: Exception) -> str:
    return f"Ollama Error: {str(error)}"
