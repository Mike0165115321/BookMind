import os

class OpenRouterConfig:
    def __init__(self):
        # OpenRouter supports multiple keys too for rotation
        self.api_keys = [
            k.strip() for k in os.getenv("OPENROUTER_API_KEYS", "").split(",")
            if k.strip()
        ]
        self.base_url = "https://openrouter.ai/api/v1"
        self.default_model = os.getenv("OPENROUTER_MODEL") # No hardcoded default
        self.temperature = float(os.getenv("OPENROUTER_TEMPERATURE", "0.3"))
        self.max_tokens = int(os.getenv("OPENROUTER_MAX_TOKENS", "4096"))
        self.site_url = os.getenv("SITE_URL", "http://localhost:8080")
        self.site_name = os.getenv("SITE_NAME", "BookMind RAG")

openrouter_config = OpenRouterConfig()
