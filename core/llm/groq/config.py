from core.config import settings

class GroqConfig:
    def __init__(self):
        self.api_keys = settings.GROQ_API_KEYS
        self.default_model = settings.GROQ_MODEL
        self.max_tokens = settings.GROQ_MAX_TOKENS
        self.temperature = settings.GROQ_TEMPERATURE

groq_config = GroqConfig()
