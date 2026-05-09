import itertools
from typing import List, Optional
from core.llm.groq.config import groq_config

class GroqKeyManager:
    def __init__(self, api_keys: List[str]):
        self.keys = api_keys
        self._key_cycler = itertools.cycle(api_keys) if api_keys else itertools.cycle([])

    def get_key(self) -> Optional[str]:
        try:
            return next(self._key_cycler)
        except StopIteration:
            return None

groq_keys = GroqKeyManager(groq_config.api_keys)
