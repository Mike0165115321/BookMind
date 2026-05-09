import json
import http.client
import time
from typing import List, Generator, Optional
from urllib.parse import urlparse

from core.llm.shared.base import BaseLLMClient
from core.llm.shared.types import Message, GenerationConfig
from core.llm.shared.response import LLMResponse, LLMStreamChunk
from core.llm.openrouter.config import openrouter_config
from core.llm.openrouter.utils import openrouter_keys

class OpenRouterClient(BaseLLMClient):
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or openrouter_config.default_model
        url = urlparse(openrouter_config.base_url)
        self.host = url.hostname
        self.port = 443 # OpenRouter is always HTTPS
        self.path_prefix = url.path

    def _get_headers(self):
        api_key = openrouter_keys.get_key()
        if not api_key:
            raise RuntimeError("❌ No API key available for OpenRouter.")
        
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": openrouter_config.site_url,
            "X-Title": openrouter_config.site_name
        }

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
            "temperature": config.temperature if config else openrouter_config.temperature,
            "max_tokens": config.max_tokens if config else openrouter_config.max_tokens
        }

        max_retries = len(openrouter_config.api_keys) or 1
        start_time = time.perf_counter()
        for attempt in range(max_retries):
            try:
                conn = http.client.HTTPSConnection(self.host, self.port)
                conn.request("POST", f"{self.path_prefix}/chat/completions", json.dumps(payload), self._get_headers())
                res = conn.getresponse()
                data = json.loads(res.read().decode())
                
                latency_ms = (time.perf_counter() - start_time) * 1000

                if "error" in data:
                    raise RuntimeError(f"OpenRouter API Error: {data['error']}")

                return LLMResponse(
                    text=data["choices"][0]["message"]["content"],
                    model_name=self.model_name,
                    provider="openrouter",
                    latency_ms=latency_ms,
                    usage={
                        "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                        "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                        "total_tokens": data.get("usage", {}).get("total_tokens", 0)
                    },
                    raw_response=data
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
        payload = {
            "model": self.model_name,
            "messages": prep_messages,
            "stream": True,
            "temperature": config.temperature if config else openrouter_config.temperature,
            "max_tokens": config.max_tokens if config else openrouter_config.max_tokens
        }

        try:
            conn = http.client.HTTPSConnection(self.host, self.port)
            conn.request("POST", f"{self.path_prefix}/chat/completions", json.dumps(payload), self._get_headers())
            res = conn.getresponse()
            
            if res.status != 200:
                error_data = res.read().decode()
                try:
                    error_json = json.loads(error_data)
                    error_msg = error_json.get("error", {}).get("message", error_data)
                except:
                    error_msg = error_data
                yield LLMStreamChunk(text=f"\n⚠️ OpenRouter API Error ({res.status}): {error_msg}")
                return

            for line in res:
                line = line.decode().strip()
                if not line:
                    continue
                    
                if line.startswith("data: "):
                    if line == "data: [DONE]":
                        break
                    try:
                        chunk_data = json.loads(line[6:])
                        content = chunk_data["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            yield LLMStreamChunk(text=content)
                    except:
                        continue
        except Exception as e:
            yield LLMStreamChunk(text=f"\n❌ OpenRouter Connection Error: {str(e)}")

    def list_models(self) -> List[str]:
        """Fetch only FREE models from OpenRouter."""
        try:
            conn = http.client.HTTPSConnection(self.host, self.port)
            conn.request("GET", f"{self.path_prefix}/models", headers=self._get_headers())
            res = conn.getresponse()
            data = json.loads(res.read().decode())
            
            # Filter for models that have ':free' in their ID, but exclude vision/moderation
            free_models = []
            for m in data.get("data", []):
                id_lower = m["id"].lower()
                if ":free" in id_lower and "vision" not in id_lower and "moderation" not in id_lower:
                    free_models.append(m["id"])
            
            return sorted(free_models)
        except Exception as e:
            print(f"⚠️ Could not list OpenRouter models: {e}")
            return []
