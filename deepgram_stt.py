"""REST client for Deepgram speech-to-text."""
from __future__ import annotations

import io
import wave
from typing import Optional

import requests

from config import DEEPGRAM_API_KEY, DEEPGRAM_LISTEN_MODEL, DEEPGRAM_STT_ENDPOINT


class DeepgramSTTError(RuntimeError):
    pass


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def transcribe_pcm(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    model: Optional[str] = None,
    language: str = "en",
) -> str:
    if not pcm_bytes:
        return ""
    if not DEEPGRAM_API_KEY:
        raise DeepgramSTTError("DEEPGRAM_API_KEY is not set.")
    wav_payload = _pcm_to_wav(pcm_bytes, sample_rate)
    params = {
        "model": model or DEEPGRAM_LISTEN_MODEL,
        "language": language,
        "smart_format": "true",
        "punctuate": "true",
    }
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav",
    }
    response = requests.post(
        DEEPGRAM_STT_ENDPOINT,
        params=params,
        headers=headers,
        data=wav_payload,
        timeout=60,
    )
    if response.status_code >= 400:
        raise DeepgramSTTError(
            f"Deepgram STT failed ({response.status_code}): {response.text}"
        )
    data = response.json()
    try:
        alternatives = data["results"]["channels"][0]["alternatives"]
        if not alternatives:
            return ""
        return (alternatives[0].get("transcript") or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise DeepgramSTTError(f"Unexpected Deepgram response: {data}") from exc
