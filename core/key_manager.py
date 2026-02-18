"""
Key Manager — Round-robin API key rotation.
Distributes API calls across multiple keys to avoid rate limits.
"""
import itertools
from typing import List
from core.config import settings


class KeyManager:
    """Manages a pool of API keys with round-robin rotation."""

    def __init__(self, api_keys: List[str], service_name: str):
        self.keys = api_keys
        self.service_name = service_name

        if not self.keys:
            print(f"⚠️  WARNING: No API keys found for {self.service_name}.")
            self._key_cycler = itertools.cycle([])
        else:
            print(f"🔑 KeyManager for {self.service_name}: {len(self.keys)} key(s) loaded.")
            self._key_cycler = itertools.cycle(self.keys)

    def get_key(self) -> str | None:
        """Get the next API key in rotation. Returns None if no keys available."""
        try:
            return next(self._key_cycler)
        except StopIteration:
            return None


# ──────────────────────────────────────────────
# Pre-built managers (import and use directly)
# ──────────────────────────────────────────────
gemini_key_manager = KeyManager(settings.GEMINI_API_KEYS, "Gemini")
groq_key_manager = KeyManager(settings.GROQ_API_KEYS, "Groq")
