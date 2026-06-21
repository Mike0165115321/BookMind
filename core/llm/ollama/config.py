import os
from urllib.parse import urlparse

class OllamaConfig:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        url = urlparse(self.base_url)
        self.port = url.port or 11434
        self.default_model = os.getenv("OLLAMA_MODEL")
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))
        self.max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "4096"))
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "2048"))  # Limit context to save memory

ollama_config = OllamaConfig()
