import json
import logging
from typing import List, Dict, Any, Optional

from core.prompts.prompt_registry import registry
from core.llm.manager import llm_manager
from core.llm.shared.types import Message, ProviderName, GenerationConfig
from core.agentic.types import EvaluationResult

logger = logging.getLogger(__name__)

def evaluate_sufficiency(
    query: str, 
    context: str, 
    agentic_provider: str, 
    agentic_model: str
) -> EvaluationResult:
    """
    Evaluate if the current context is sufficient to answer the original query.
    Uses LLMManager with configurable provider/model.
    """
    try:
        # 1. Resolve Provider & Model
        try:
            p_enum = ProviderName(agentic_provider.lower())
        except (ValueError, AttributeError):
            p_enum = ProviderName.GROQ
            
        m_name = agentic_model

        user_prompt = f"""คำถามต้นฉบับ: {query}
---
ข้อมูลที่ค้นหามาได้:
{context}
---
ประเมินว่าข้อมูลเพียงพอต่อการตอบคำถามต้นฉบับหรือไม่"""

        # 2. Call LLM via Manager (Explicit keywords)
        messages = [
            Message(role="system", content=registry.get("agentic_eval")),
            Message(role="user", content=user_prompt),
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
        
        result = EvaluationResult(
            is_sufficient=parsed.get("is_sufficient", False),
            confidence=parsed.get("confidence", 0.0),
            missing_aspects=parsed.get("missing_aspects", []),
            follow_up_queries=parsed.get("follow_up_queries", []),
            reasoning=parsed.get("reasoning", "")
        )
        
        return result

    except Exception as e:
        print(f"   ⚠️  Evaluation failed ({e}), assuming insufficient")
        return EvaluationResult(
            is_sufficient=False,
            confidence=0.0,
            missing_aspects=[f"Error during evaluation: {e}"],
            follow_up_queries=[query]
        )
