"""
Query Decomposer — Breaks complex questions into targeted sub-queries.

Uses Groq LLM to analyze query complexity and decompose:
  - Simple query (single topic/book) → pass through unchanged
  - Complex query (multi-topic/comparison) → split into sub-queries

This is the "brain" that decides HOW to search, not just WHAT to search.
Works alongside HyDE — HyDE improves search quality per query,
Decomposer improves search COVERAGE across topics.

Example:
  Input:  "เปรียบเทียบหลักการลงทุนจาก Rich Dad กับ Psychology of Money"
  Output: [
      "หลักการลงทุน Rich Dad Poor Dad",
      "หลักการลงทุน Psychology of Money"
  ]
"""
import json
from dataclasses import dataclass, field
from groq import Groq
from core.config import settings
from core.key_manager import groq_key_manager
from core.prompts.prompt_registry import registry


# ──────────────────────────────────────────────
# Result Data Structure
# ──────────────────────────────────────────────
@dataclass
class DecompositionResult:
    """Result of query decomposition analysis."""
    query_type: str             # "simple" or "complex"
    sub_queries: list[str]      # List of sub-queries (1 for simple, N for complex)
    reasoning: str = ""         # Why the LLM decomposed this way
    original_query: str = ""    # Original query for reference


# ──────────────────────────────────────────────
# Decomposition Prompt
# ──────────────────────────────────────────────
# Prompt is now loaded via registry.get("agentic_decompose")


def _get_groq_client() -> Groq:
    """Create a Groq client with the next API key from rotation."""
    api_key = groq_key_manager.get_key()
    if not api_key:
        raise RuntimeError("❌ ไม่มี API key สำหรับ Groq — กรุณาตั้งค่าใน .env")
    return Groq(api_key=api_key)


def decompose(query: str) -> DecompositionResult:
    """
    Analyze query complexity and decompose into sub-queries.

    For simple queries (single topic): returns the query unchanged.
    For complex queries (multi-topic/comparison): splits into targeted sub-queries.

    Args:
        query: Original user query

    Returns:
        DecompositionResult with query_type, sub_queries, and reasoning
    """
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": registry.get("agentic_decompose")},
                {"role": "user", "content": f"คำถาม: {query}"},
            ],
            max_tokens=256,
            temperature=0.2,  # Low temp for consistent structured output
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        query_type = parsed.get("query_type", "simple")
        sub_queries = parsed.get("sub_queries", [query])
        reasoning = parsed.get("reasoning", "")

        # Safety: ensure at least 1 sub-query
        if not sub_queries:
            sub_queries = [query]

        # Safety: if only 1 sub-query, treat as simple
        if len(sub_queries) == 1:
            query_type = "simple"

        result = DecompositionResult(
            query_type=query_type,
            sub_queries=sub_queries,
            reasoning=reasoning,
            original_query=query,
        )

        print(f"   🔀 Decompose: {query_type} → {len(sub_queries)} sub-queries")
        for i, sq in enumerate(sub_queries, 1):
            print(f"      [{i}] {sq}")

        return result

    except Exception as e:
        print(f"   ⚠️  Decompose failed ({e}), using original query")
        return DecompositionResult(
            query_type="simple",
            sub_queries=[query],
            reasoning=f"Fallback due to error: {e}",
            original_query=query,
        )
"""
"""
