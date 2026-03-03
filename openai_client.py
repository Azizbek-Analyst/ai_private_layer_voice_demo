"""OpenAI chat helper for the insurance voice agent."""
from __future__ import annotations

from typing import Dict, List, Optional

from config import OPENAI_API_KEY, OPENAI_MODEL
from insurance_prompt import build_system_prompt


class OpenAIUnavailable(RuntimeError):
    pass


_client = None
_system_prompt = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not OPENAI_API_KEY:
        raise OpenAIUnavailable("OPENAI_API_KEY is not set.")
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - import guard
        raise OpenAIUnavailable("Install openai: pip install openai") from exc
    _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def generate_response(
    conversation: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.4,
    max_tokens: int = 600,
) -> str:
    client = _get_client()
    global _system_prompt
    if _system_prompt is None:
        _system_prompt = build_system_prompt()
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _system_prompt},
    ]
    messages.extend(conversation)
    response = client.chat.completions.create(
        model=model or OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()
