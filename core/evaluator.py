"""
Evaluator — Assesses information sufficiency for the Agentic RAG loop.

After each search iteration, the Evaluator checks:
  1. Does the gathered information cover all aspects of the query?
  2. What's missing? What follow-up queries could fill the gaps?
  3. Confidence score: 0.0 (nothing useful) → 1.0 (fully sufficient)

This is the "quality gate" that decides when to STOP searching.
Without this, the agent would either search too little or loop forever.

Decision Logic:
  confidence >= threshold → STOP, generate answer
  confidence < threshold  → CONTINUE, search with follow-up queries
  max_iterations reached  → STOP regardless (hard limit)
"""
import json
from dataclasses import dataclass, field
from groq import Groq
from core.config import settings
from core.key_manager import groq_key_manager


# ──────────────────────────────────────────────
# Result Data Structure
# ──────────────────────────────────────────────
@dataclass
class EvaluationResult:
    """Result of sufficiency evaluation."""
    is_sufficient: bool             # True if enough info to answer well
    confidence: float               # 0.0 - 1.0 confidence score
    missing_aspects: list[str]      # What's still missing
    follow_up_queries: list[str]    # Suggested queries to fill gaps
    reasoning: str = ""             # Why this evaluation


# ──────────────────────────────────────────────
# Evaluation Prompt
# ──────────────────────────────────────────────
EVALUATE_SYSTEM_PROMPT = """คุณเป็น AI ที่ประเมินว่าข้อมูลที่ค้นหาได้เพียงพอต่อการตอบคำถามหรือไม่

คุณจะได้รับ:
1. คำถามต้นฉบับ
2. Sub-queries ที่วางแผนจะค้นหา
3. สรุปข้อมูลที่ค้นได้แล้ว

ภารกิจ: ประเมินว่าข้อมูลเพียงพอหรือยัง

กฎการประเมิน:
- confidence 0.8-1.0 = ข้อมูลครบ ตอบได้ดี
- confidence 0.5-0.79 = ข้อมูลพอใช้ได้ แต่ยังมีส่วนที่ขาด
- confidence 0.0-0.49 = ข้อมูลยังไม่เพียงพอ

- ถ้ายังไม่เพียงพอ ให้แนะนำ follow_up_queries (1-2 อัน) ที่จะช่วยเติมส่วนที่ขาด
- follow_up_queries ต้องต่างจาก sub_queries เดิม (ไม่ใช่ค้นซ้ำ)

ตอบเป็น JSON เท่านั้น:
{
  "is_sufficient": true/false,
  "confidence": 0.0-1.0,
  "missing_aspects": ["สิ่งที่ยังขาด..."],
  "follow_up_queries": ["query เพิ่มเติม..."],
  "reasoning": "เหตุผลสั้นๆ"
}"""


def _get_groq_client() -> Groq:
    """Create a Groq client with the next API key from rotation."""
    api_key = groq_key_manager.get_key()
    if not api_key:
        raise RuntimeError("❌ ไม่มี API key สำหรับ Groq — กรุณาตั้งค่าใน .env")
    return Groq(api_key=api_key)


def evaluate_sufficiency(
    original_query: str,
    sub_queries: list[str],
    context_summary: str,
    threshold: float = 0.7,
) -> EvaluationResult:
    """
    Evaluate whether gathered information is sufficient to answer the query.

    Uses Groq LLM to analyze coverage of the original query
    against the information gathered so far.

    Args:
        original_query: The user's original question
        sub_queries: List of sub-queries that were planned
        context_summary: Summary of gathered chunks from AgentMemory
        threshold: Confidence threshold for sufficiency

    Returns:
        EvaluationResult with is_sufficient, confidence, and follow-up queries
    """
    try:
        client = _get_groq_client()

        user_prompt = f"""คำถามต้นฉบับ: {original_query}

Sub-queries ที่วางแผนไว้: {json.dumps(sub_queries, ensure_ascii=False)}

ข้อมูลที่ค้นหาได้:
{context_summary}

ประเมินว่าข้อมูลเพียงพอต่อการตอบคำถามต้นฉบับหรือไม่"""

        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": EVALUATE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=256,
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        confidence = float(parsed.get("confidence", 0.5))
        is_sufficient = confidence >= threshold
        missing = parsed.get("missing_aspects", [])
        follow_ups = parsed.get("follow_up_queries", [])
        reasoning = parsed.get("reasoning", "")

        result = EvaluationResult(
            is_sufficient=is_sufficient,
            confidence=confidence,
            missing_aspects=missing,
            follow_up_queries=follow_ups,
            reasoning=reasoning,
        )

        status = "✅ เพียงพอ" if is_sufficient else "🔄 ยังไม่ครบ"
        print(f"   📊 Evaluate: {status} (confidence={confidence:.2f}, threshold={threshold})")
        if missing:
            for m in missing:
                print(f"      ❌ ขาด: {m}")
        if follow_ups:
            for fq in follow_ups:
                print(f"      🔍 ค้นเพิ่ม: {fq}")

        return result

    except Exception as e:
        print(f"   ⚠️  Evaluate failed ({e}), assuming sufficient")
        return EvaluationResult(
            is_sufficient=True,
            confidence=0.5,
            missing_aspects=[],
            follow_up_queries=[],
            reasoning=f"Fallback due to error: {e}",
        )
"""
"""
