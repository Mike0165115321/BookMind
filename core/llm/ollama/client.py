import json
import http.client
import time
from typing import List, Generator, Optional
from urllib.parse import urlparse

from core.llm.shared.base import BaseLLMClient
from core.llm.shared.types import Message, GenerationConfig
from core.llm.shared.response import LLMResponse, LLMStreamChunk
from core.llm.ollama.config import ollama_config

class OllamaClient(BaseLLMClient):
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or ollama_config.default_model
        url = urlparse(ollama_config.base_url)
        self.host = url.hostname
        self.port = url.port or (80 if url.scheme == 'http' else 443)

    def _prepare_messages(self, messages: List[Message]):
        return [{"role": m.role, "content": m.content} for m in messages]

    def generate(
        self, 
        messages: List[Message], 
        config: Optional[GenerationConfig] = None
    ) -> LLMResponse:
        prep_messages = self._prepare_messages(messages)
        payload = {
            "model": self.model_name,
            "messages": prep_messages,
            "stream": False,
            "options": {
                "temperature": config.temperature if config else ollama_config.temperature,
                "num_predict": config.max_tokens if config else ollama_config.max_tokens
            }
        }

        start_time = time.perf_counter()
        try:
            conn = http.client.HTTPConnection(self.host, self.port)
            conn.request("POST", "/api/chat", json.dumps(payload), {"Content-Type": "application/json"})
            res = conn.getresponse()
            data = json.loads(res.read().decode())
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return LLMResponse(
                text=data["message"]["content"],
                model_name=self.model_name,
                provider="ollama",
                latency_ms=latency_ms,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                },
                raw_response=data
            )
        except Exception as e:
            raise RuntimeError(f"Ollama connection failed: {e}")

    def generate_stream(
        self, 
        messages: List[Message], 
        config: Optional[GenerationConfig] = None
    ) -> Generator[LLMStreamChunk, None, None]:
        prep_messages = self._prepare_messages(messages)
        payload = {
            "model": self.model_name,
            "messages": prep_messages,
            "stream": True,
            "options": {
                "temperature": config.temperature if config else ollama_config.temperature,
                "num_predict": config.max_tokens if config else ollama_config.max_tokens
            }
        }

        try:
            conn = http.client.HTTPConnection(self.host, self.port)
            conn.request("POST", "/api/chat", json.dumps(payload), {"Content-Type": "application/json"})
            res = conn.getresponse()
            
            for line in res:
                if line:
                    chunk_data = json.loads(line.decode())
                    if "message" in chunk_data:
                        yield LLMStreamChunk(text=chunk_data["message"]["content"])
                    if chunk_data.get("done"):
                        break
        except Exception as e:
            yield LLMStreamChunk(text=f"\n❌ Ollama Error: {str(e)}", is_last=True)

    def list_models(self) -> List[str]:
        """Fetch available models from the local Ollama service."""
        try:
            conn = http.client.HTTPConnection(self.host, self.port)
            conn.request("GET", "/api/tags")
            res = conn.getresponse()
            data = json.loads(res.read().decode())
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            print(f"⚠️ Could not list Ollama models: {e}")
            return []
