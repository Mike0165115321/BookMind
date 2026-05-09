from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class LLMResponse:
    text: str
    model_name: str
    provider: str
    latency_ms: float = 0.0
    usage: Dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    finish_reason: Optional[str] = None
    raw_response: Any = None # Original response from the provider SDK

@dataclass
class LLMStreamChunk:
    text: str
    is_last: bool = False
    usage: Optional[Dict[str, int]] = None
