import time
from typing import List, Generator, Optional
from google import genai
from google.genai import types, errors

from core.llm.shared.base import BaseLLMClient
from core.llm.shared.types import Message, GenerationConfig
from core.llm.shared.response import LLMResponse, LLMStreamChunk
from core.llm.gemini.config import gemini_config
from core.llm.gemini.utils import gemini_keys

class GeminiClient(BaseLLMClient):
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or gemini_config.default_model

    def _get_client(self) -> genai.Client:
        api_key = gemini_keys.get_key()
        if not api_key:
            raise RuntimeError("❌ No API key available for Gemini.")
        return genai.Client(api_key=api_key)

    def _prepare_contents(self, messages: List[Message]):
        """Convert shared Message format to Gemini format."""
        # Note: Gemini's latest SDK often separates system_instruction
        system_instruction = None
        contents = []
        
        for m in messages:
            if m.role == 'system':
                system_instruction = m.content
            else:
                contents.append({"role": m.role, "parts": [{"text": m.content}]})
        
        return contents, system_instruction

    def generate(
        self, 
        messages: List[Message], 
        config: Optional[GenerationConfig] = None
    ) -> LLMResponse:
        contents, system_instruction = self._prepare_contents(messages)
        temp = config.temperature if config else gemini_config.temperature
        
        gen_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temp,
            max_output_tokens=config.max_tokens if config else gemini_config.max_tokens
        )

        max_retries = len(gemini_config.api_keys) or 1
        start_time = time.perf_counter()
        for attempt in range(max_retries):
            try:
                client = self._get_client()
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=gen_config,
                )
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                # Calculate usage (if available)
                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                if hasattr(response, 'usage_metadata'):
                    usage = {
                        "prompt_tokens": response.usage_metadata.prompt_token_count,
                        "completion_tokens": response.usage_metadata.candidates_token_count,
                        "total_tokens": response.usage_metadata.total_token_count
                    }

                return LLMResponse(
                    text=response.text,
                    model_name=self.model_name,
                    provider="gemini",
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
        contents, system_instruction = self._prepare_contents(messages)
        temp = config.temperature if config else gemini_config.temperature
        
        gen_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temp,
            max_output_tokens=config.max_tokens if config else gemini_config.max_tokens
        )

        max_retries = len(gemini_config.api_keys) or 1
        for attempt in range(max_retries):
            try:
                client = self._get_client()
                for chunk in client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=gen_config,
                ):
                    if chunk.text:
                        yield LLMStreamChunk(text=chunk.text)
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                yield LLMStreamChunk(text=f"\n❌ Error: {str(e)}", is_last=True)
                return

    def list_models(self) -> List[str]:
        """List available Google Gemini models."""
        try:
            client = self._get_client()
            models = client.models.list()
            return [m.name for m in models if "generateContent" in m.supported_generation_methods]
        except Exception as e:
            print(f"⚠️ Could not list Gemini models: {e}")
            return []
