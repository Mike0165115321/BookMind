import os

class OllamaConfig:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.default_model = os.getenv("OLLAMA_MODEL") # No more hardcoded fallback
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))
        self.max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "4096"))

ollama_config = OllamaConfig()
