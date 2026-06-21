import json
import http.client
import time
import os
from typing import List, Generator, Optional
from urllib.parse import urlparse

from core.llm.shared.base import BaseLLMClient
from core.llm.shared.types import Message, GenerationConfig
from core.llm.shared.response import LLMResponse, LLMStreamChunk
from core.llm.ollama.config import ollama_config

class OllamaClient(BaseLLMClient):
    _cached_host = None

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or ollama_config.default_model
        self.port = ollama_config.port # Default 11434
        self.host = self._get_best_host()

    def _get_best_host(self) -> str:
        """Try to discover the reachable Ollama host (Local, WSL Host, or Docker Host)."""
        # 1. Check if we already found a working host
        if OllamaClient._cached_host:
            return OllamaClient._cached_host

        # 2. Potential hosts to try
        potential_hosts = ["localhost", "127.0.0.1"]
        
        # Try to find Windows Host IP from WSL using 'ip route'
        try:
            import subprocess
            cmd = "ip route show | grep default | awk '{print $3}'"
            gw_ip = subprocess.check_output(cmd, shell=True).decode().strip()
            if gw_ip:
                potential_hosts.append(gw_ip)
        except:
            pass

        # Fallback to nameserver if 'ip route' fails
        try:
            if os.path.exists("/etc/resolv.conf"):
                with open("/etc/resolv.conf", "r") as f:
                    for line in f:
                        if "nameserver" in line:
                            ns_ip = line.split()[1].strip()
                            if ns_ip not in ["8.8.8.8", "1.1.1.1", "8.8.4.4"]: # Skip public DNS
                                potential_hosts.append(ns_ip)
        except:
            pass
            
        potential_hosts.append("host.docker.internal")

        # 3. Test each host
        print(f"🔍 Ollama Discovery: Testing hosts {potential_hosts}...")
        for host in potential_hosts:
            try:
                # Use a slightly longer timeout and be explicit about the host being tested
                conn = http.client.HTTPConnection(host, self.port, timeout=1.0)
                conn.request("GET", "/api/tags")
                res = conn.getresponse()
                if res.status == 200:
                    print(f"✅ Auto-discovered Ollama host: {host}")
                    OllamaClient._cached_host = host
                    return host
            except Exception as e:
                # print(f"  ❌ Host {host} failed: {e}") # Keep it quiet unless we need deep debug
                continue

        # 4. Fallback to localhost if all else fails
        return "localhost"

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
                "num_predict": config.max_tokens if config else ollama_config.max_tokens,
                "num_ctx": ollama_config.num_ctx
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
                "num_predict": config.max_tokens if config else ollama_config.max_tokens,
                "num_ctx": ollama_config.num_ctx
            }
        }

        try:
            conn = http.client.HTTPConnection(self.host, self.port)
            conn.request("POST", "/api/chat", json.dumps(payload), {"Content-Type": "application/json"})
            res = conn.getresponse()
            
            if res.status != 200:
                error_text = res.read().decode()
                try:
                    err_json = json.loads(error_text)
                    msg = err_json.get("error", error_text)
                except:
                    msg = error_text
                yield LLMStreamChunk(text=f"\n⚠️ Ollama API Error ({res.status}): {msg}", is_last=True)
                return

            for line in res:
                if line:
                    try:
                        chunk_data = json.loads(line.decode())
                        if "error" in chunk_data:
                            yield LLMStreamChunk(text=f"\n⚠️ Ollama Error: {chunk_data['error']}", is_last=True)
                            break
                        if "message" in chunk_data:
                            yield LLMStreamChunk(text=chunk_data["message"]["content"])
                        if chunk_data.get("done"):
                            break
                    except Exception as parse_err:
                        continue
        except Exception as e:
            yield LLMStreamChunk(text=f"\n❌ Ollama Error: {str(e)}", is_last=True)

    def list_models(self) -> List[str]:
        """Fetch available models from the local Ollama service."""
        try:
            conn = http.client.HTTPConnection(self.host, self.port)
            conn.request("GET", "/api/tags")
            res = conn.getresponse()
            data = json.loads(res.read().decode())
            
            models = []
            for m in data.get("models", []):
                name = m["name"].lower()
                # Exclude embedding models which cannot be used for text generation
                if "embed" not in name and "bert" not in name:
                    models.append(m["name"])
                    
            return sorted(models)
        except Exception as e:
            print(f"⚠️ Could not list Ollama models: {e}")
            return []
