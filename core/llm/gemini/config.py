from core.config import settings

class GeminiConfig:
    def __init__(self):
        self.api_keys = settings.GEMINI_API_KEYS
        self.default_model = settings.GEMINI_MODEL
        self.max_tokens = settings.GEMINI_MAX_TOKENS
        self.temperature = settings.GEMINI_TEMPERATURE

gemini_config = GeminiConfig()
