"""
Core Settings — loads secrets from .env file.
Provides parsed API keys and LLM model settings.
Separate from config.py (RAG-specific tuning) for Separation of Concerns.
"""
import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))


class Settings:
    """Centralized settings loaded from environment variables."""

    # ──────────────────────────────────────────────
    # API Keys (parsed from comma-separated .env values)
    # ──────────────────────────────────────────────
    GEMINI_API_KEYS: list[str] = [
        k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",")
        if k.strip()
    ]
    GROQ_API_KEYS: list[str] = [
        k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",")
        if k.strip()
    ]

    # ──────────────────────────────────────────────
    # Gemini (LLM Generation)
    # ──────────────────────────────────────────────
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_MAX_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "4096"))
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))

    # ──────────────────────────────────────────────
    # Groq (Query Transform / HyDE)
    # ──────────────────────────────────────────────
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "512"))
    GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.7"))


# Singleton instance — import this everywhere
settings = Settings()
