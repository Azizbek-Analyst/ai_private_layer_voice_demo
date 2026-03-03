"""REST client for Deepgram text-to-speech."""
from __future__ import annotations

from typing import Optional

import requests

from config import (
    DEEPGRAM_API_KEY,
    DEEPGRAM_SPEAK_MODEL,
    DEEPGRAM_TTS_ENCODING,
    DEEPGRAM_TTS_ENDPOINT,
    DEEPGRAM_TTS_SAMPLE_RATE,
)


class DeepgramTTSError(RuntimeError):
    pass


def synthesize(
    text: str,
    *,
    model: Optional[str] = None,
    sample_rate: int = DEEPGRAM_TTS_SAMPLE_RATE,
    encoding: str = DEEPGRAM_TTS_ENCODING,
) -> bytes:
    if not text:
        return b""
    if not DEEPGRAM_API_KEY:
        raise DeepgramTTSError("DEEPGRAM_API_KEY is not set.")
    params = {
        "model": model or DEEPGRAM_SPEAK_MODEL,
        "encoding": encoding,
        "sample_rate": str(sample_rate),
    }
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "audio/linear16",
    }
    response = requests.post(
        DEEPGRAM_TTS_ENDPOINT,
        params=params,
        headers=headers,
        json={"text": text},
        timeout=60,
    )
    if response.status_code >= 400:
        raise DeepgramTTSError(
            f"Deepgram TTS failed ({response.status_code}): {response.text}"
        )
    return response.content
