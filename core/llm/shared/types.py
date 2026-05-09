from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any

class ProviderName(str, Enum):
    GEMINI = "gemini"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    GROQ = "groq"

@dataclass
class GenerationConfig:
    temperature: float = 0.3
    max_tokens: Optional[int] = 4096
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: List[str] = field(default_factory=list)

@dataclass
class Message:
    role: str # 'system', 'user', 'assistant'
    content: str
