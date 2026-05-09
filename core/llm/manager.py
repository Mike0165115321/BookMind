import logging
from typing import List, Optional, Generator, Dict, Any

from core.llm.shared.types import Message, GenerationConfig, ProviderName
from core.llm.shared.response import LLMResponse, LLMStreamChunk
from core.llm.shared.base import BaseLLMClient

# Providers
from core.llm.gemini.client import GeminiClient
from core.llm.groq.client import GroqClient
from core.llm.ollama.client import OllamaClient
from core.llm.openrouter.client import OpenRouterClient

logger = logging.getLogger(__name__)

class LLMManager:
    def __init__(self):
        # Cache instances of clients
        self._clients: Dict[ProviderName, BaseLLMClient] = {}
        
        # Mapping of provider names to their client classes
        self._client_registry = {
            ProviderName.GEMINI: GeminiClient,
            ProviderName.GROQ: GroqClient,
            ProviderName.OLLAMA: OllamaClient,
            ProviderName.OPENROUTER: OpenRouterClient
        }

    def _get_client(self, provider: ProviderName, model_name: Optional[str] = None) -> BaseLLMClient:
        """Lazy load and return the appropriate client."""
        if provider not in self._client_registry:
            raise ValueError(f"❌ Provider '{provider}' is not supported.")
        
        # We can cache by provider+model if needed, but for now simple provider cache
        if provider not in self._clients:
            client_class = self._client_registry[provider]
            self._clients[provider] = client_class(model_name=model_name)
        elif model_name:
            # If a specific model is requested, update the cached client's model
            self._clients[provider].model_name = model_name
            
        return self._clients[provider]

    def generate(
        self, 
        provider: ProviderName,
        messages: List[Message],
        model_name: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> LLMResponse:
        """Dispatcher for non-streaming generation."""
        client = self._get_client(provider, model_name)
        return client.generate(messages, config)

    def generate_stream(
        self,
        provider: ProviderName,
        messages: List[Message],
        model_name: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> Generator[LLMStreamChunk, None, None]:
        """Dispatcher for streaming generation."""
        client = self._get_client(provider, model_name)
        return client.generate_stream(messages, config)

    def list_models(self, provider: ProviderName) -> List[str]:
        """List available models for a specific provider."""
        try:
            client = self._get_client(provider)
            return client.list_models()
        except Exception as e:
            logger.error(f"Error listing models for {provider}: {e}")
            return []

    def get_all_available_models(self) -> Dict[str, List[str]]:
        """Utility to get models from all registered providers."""
        all_models = {}
        for p in self._client_registry.keys():
            all_models[p.value] = self.list_models(p)
        return all_models

# Singleton instance
llm_manager = LLMManager()
