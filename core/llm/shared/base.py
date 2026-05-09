from abc import ABC, abstractmethod
from typing import List, Generator, Optional
from core.llm.shared.types import Message, GenerationConfig
from core.llm.shared.response import LLMResponse, LLMStreamChunk

class BaseLLMClient(ABC):
    @abstractmethod
    def generate(
        self, 
        messages: List[Message], 
        config: Optional[GenerationConfig] = None
    ) -> LLMResponse:
        pass

    @abstractmethod
    def generate_stream(
        self, 
        messages: List[Message], 
        config: Optional[GenerationConfig] = None
    ) -> Generator[LLMStreamChunk, None, None]:
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models for this provider."""
        pass
