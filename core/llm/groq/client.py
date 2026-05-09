import time
from typing import List, Generator, Optional
from groq import Groq

from core.llm.shared.base import BaseLLMClient
from core.llm.shared.types import Message, GenerationConfig
from core.llm.shared.response import LLMResponse, LLMStreamChunk
from core.llm.groq.config import groq_config
from core.llm.groq.utils import groq_keys

class GroqClient(BaseLLMClient):
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or groq_config.default_model

    def _get_client(self) -> Groq:
        api_key = groq_keys.get_key()
        if not api_key:
            raise RuntimeError("❌ No API key available for Groq.")
        return Groq(api_key=api_key)

    def _prepare_messages(self, messages: List[Message]):
        return [{"role": m.role, "content": m.content} for m in messages]

    def generate(
        self, 
        messages: List[Message], 
        config: Optional[GenerationConfig] = None
    ) -> LLMResponse:
        prep_messages = self._prepare_messages(messages)
        temp = config.temperature if config else groq_config.temperature
        
        max_retries = len(groq_config.api_keys) or 1
        start_time = time.perf_counter()
        for attempt in range(max_retries):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=prep_messages,
                    temperature=temp,
                    max_tokens=config.max_tokens if config else groq_config.max_tokens
                )
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

                return LLMResponse(
                    text=response.choices[0].message.content,
                    model_name=self.model_name,
                    provider="groq",
                    latency_ms=latency_ms,
                    usage=usage,
                    raw_response=response
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise e

    def generate_stream(
        self, 
        messages: List[Message], 
        config: Optional[GenerationConfig] = None
    ) -> Generator[LLMStreamChunk, None, None]:
        prep_messages = self._prepare_messages(messages)
        temp = config.temperature if config else groq_config.temperature
        
        max_retries = len(groq_config.api_keys) or 1
        for attempt in range(max_retries):
            try:
                client = self._get_client()
                stream = client.chat.completions.create(
                    model=self.model_name,
                    messages=prep_messages,
                    temperature=temp,
                    max_tokens=config.max_tokens if config else groq_config.max_tokens,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield LLMStreamChunk(text=chunk.choices[0].delta.content)
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                yield LLMStreamChunk(text=f"\n❌ Groq Error: {str(e)}", is_last=True)
                return

    def list_models(self) -> List[str]:
        """List available Groq models."""
        try:
            client = self._get_client()
            models = client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            print(f"⚠️ Could not list Groq models: {e}")
            return []
