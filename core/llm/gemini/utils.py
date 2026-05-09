import itertools
from typing import List, Optional
from core.llm.gemini.config import gemini_config

class GeminiKeyManager:
    def __init__(self, api_keys: List[str]):
        self.keys = api_keys
        self._key_cycler = itertools.cycle(api_keys) if api_keys else itertools.cycle([])

    def get_key(self) -> Optional[str]:
        try:
            return next(self._key_cycler)
        except StopIteration:
            return None

gemini_keys = GeminiKeyManager(gemini_config.api_keys)

def map_gemini_error(error: Exception) -> str:
    """Map Google GenAI errors to user-friendly messages."""
    # This can be expanded based on specific error types from google.genai
    return f"Gemini Error: {str(error)}"
