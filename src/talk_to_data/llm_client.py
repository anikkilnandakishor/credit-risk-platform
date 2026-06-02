"""LLM client abstraction for OpenAI and Google Gemini."""

from __future__ import annotations

from typing import Literal

from src.utils.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

Provider = Literal["openai", "gemini"]


class LLMServiceError(Exception):
    """Base error for LLM API failures."""


class LLMQuotaError(LLMServiceError):
    """Rate limit, quota, or billing errors."""


class LLMNotConfiguredError(LLMServiceError):
    """No API key available."""


def has_llm_configured() -> bool:
    return bool(OPENAI_API_KEY or GEMINI_API_KEY)


def get_active_provider() -> Provider:
    """Resolve LLM provider from config and available API keys."""
    provider = (LLM_PROVIDER or "openai").lower()
    if provider == "gemini" and GEMINI_API_KEY:
        return "gemini"
    if provider == "openai" and OPENAI_API_KEY:
        return "openai"
    if GEMINI_API_KEY:
        return "gemini"
    if OPENAI_API_KEY:
        return "openai"
    raise LLMNotConfiguredError(
        "No LLM API key configured. Set OPENAI_API_KEY or GEMINI_API_KEY in .env"
    )


def _is_quota_or_rate_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    indicators = (
        "quota",
        "rate limit",
        "rate_limit",
        "429",
        "insufficient",
        "billing",
        "exceeded",
        "resource_exhausted",
        "too many requests",
    )
    return any(x in msg for x in indicators)


def complete(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    """Run a chat completion with the configured provider."""
    try:
        provider = get_active_provider()
    except LLMNotConfiguredError:
        raise
    try:
        if provider == "openai":
            return _complete_openai(system_prompt, user_prompt, temperature)
        return _complete_gemini(system_prompt, user_prompt, temperature)
    except Exception as exc:
        if _is_quota_or_rate_error(exc):
            raise LLMQuotaError(str(exc)) from exc
        raise LLMServiceError(str(exc)) from exc


def _complete_openai(system_prompt: str, user_prompt: str, temperature: float) -> str:
    if not OPENAI_API_KEY:
        raise LLMNotConfiguredError("OPENAI_API_KEY is not set.")
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    text = response.choices[0].message.content or ""
    logger.info("OpenAI completion (%s chars)", len(text))
    return text.strip()


def _complete_gemini(system_prompt: str, user_prompt: str, temperature: float) -> str:
    if not GEMINI_API_KEY:
        raise LLMNotConfiguredError("GEMINI_API_KEY is not set.")
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        system_instruction=system_prompt,
    )
    response = model.generate_content(
        user_prompt,
        generation_config=genai.types.GenerationConfig(temperature=temperature),
    )
    text = response.text or ""
    logger.info("Gemini completion (%s chars)", len(text))
    return text.strip()
