import json
import logging
from typing import List, Optional
from dataclasses import dataclass

from core.prompts.prompt_registry import registry
from core.llm.manager import llm_manager
from core.llm.shared.types import Message, ProviderName, GenerationConfig
from core.agentic.types import DecompositionResult

logger = logging.getLogger(__name__)

def decompose(query: str, agentic_provider: str, agentic_model: str) -> DecompositionResult:
    """
    Decompose a complex query into simpler sub-queries.
    Uses LLMManager with configurable provider/model.
    """
    try:
        # 1. Resolve Provider & Model
        try:
            p_enum = ProviderName(agentic_provider.lower())
        except (ValueError, AttributeError):
            p_enum = ProviderName.GROQ
            
        m_name = agentic_model

        # 2. Call LLM via Manager (Explicitly use keywords to avoid issues)
        messages = [
            Message(role="system", content=registry.get("agentic_decompose")),
            Message(role="user", content=f"คำถาม: {query}"),
        ]
        config = GenerationConfig(temperature=0.2)
        
        response = llm_manager.generate(
            provider=p_enum,
            messages=messages,
            model_name=m_name,
            config=config
        )
        response_text = response.text

        # 3. Parse JSON Result
        parsed = json.loads(response_text)
        query_type = parsed.get("query_type", "simple")
        sub_queries = parsed.get("sub_queries", [query])
        reasoning = parsed.get("reasoning", "")

        result = DecompositionResult(
            query_type=query_type,
            sub_queries=sub_queries,
            reasoning=reasoning,
            original_query=query,
        )

        print(f"   🔀 Decompose: {query_type} → {len(sub_queries)} sub-queries (via {p_enum}:{m_name})")
        return result

    except Exception as e:
        print(f"   ⚠️  Decompose failed ({e}), using original query")
        return DecompositionResult(
            query_type="simple",
            sub_queries=[query],
            reasoning=f"Fallback due to error: {e}",
            original_query=query,
        )
