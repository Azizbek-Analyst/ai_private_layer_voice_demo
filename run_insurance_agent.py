"""Insurance voice agent: Deepgram STT/TTS + Private Layer + OpenAI."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from audio_utils import INPUT_SAMPLE_RATE, play_pcm16, record_until_pause
from config import (
    DEEPGRAM_API_KEY,
    DEEPGRAM_LISTEN_MODEL,
    DEEPGRAM_SPEAK_MODEL,
    DEEPGRAM_TTS_SAMPLE_RATE,
    LOGS_DIR,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    PRIVATE_LAYER_API_KEY,
    PRIVATE_LAYER_DETECT_PATH,
    PRIVATE_LAYER_TENANT,
    PRIVATE_LAYER_URL,
)
from deepgram_stt import DeepgramSTTError, transcribe_pcm
from deepgram_tts import DeepgramTTSError, synthesize
from insurance_prompt import build_system_prompt, get_greeting
from openai_client import OpenAIUnavailable, generate_response
from policy_storage import PolicyUpdate, save_policy_update
from private_layer_client import PrivateLayerClient, PrivateLayerError

EXIT_PHRASES = {"bye", "goodbye", "quit", "exit", "cancel", "stop", "thanks"}


def _parse_policy_line(line: str) -> Optional[PolicyUpdate]:
    marker = "POLICY_UPDATE"
    if marker not in line.upper():
        return None
    try:
        payload = line.split(":", 1)[1].strip()
        data = json.loads(payload)
    except (IndexError, json.JSONDecodeError):
        return None
    required_fields = [
        "policy_number",
        "old_name",
        "new_name",
        "date_of_changes",
        "phone_number",
    ]
    if any(not str(data.get(field, "")).strip() for field in required_fields):
        return None
    return PolicyUpdate(
        policy_number=str(data.get("policy_number", "")).strip(),
        old_name=str(data.get("old_name", "")).strip(),
        new_name=str(data.get("new_name", "")).strip(),
        date_of_changes=str(data.get("date_of_changes", "")).strip(),
        phone_number=str(data.get("phone_number", "")).strip(),
        details=str(data.get("details", "")).strip(),
        raw=line.strip(),
    )


def _should_exit(text: str) -> bool:
    return text.strip().lower() in EXIT_PHRASES


def _speak(text: str, log_event) -> None:
    if not text:
        return
    try:
        audio = synthesize(text, model=DEEPGRAM_SPEAK_MODEL, sample_rate=DEEPGRAM_TTS_SAMPLE_RATE)
        play_pcm16(audio, sample_rate=DEEPGRAM_TTS_SAMPLE_RATE)
    except DeepgramTTSError as exc:
        print("[TTS Error]", exc)
        log_event(f"TTS_ERROR: {exc}")
        print(text)


def _log_policy(update: PolicyUpdate, log_event) -> None:
    path = save_policy_update(update)
    print(f"[Storage] Policy update saved to {path}")
    log_event(
        "Policy update saved: policy_number={policy}, old={old}, new={new}, date={date}, phone={phone}".format(
            policy=update.policy_number,
            old=update.old_name,
            new=update.new_name,
            date=update.date_of_changes,
            phone=update.phone_number,
        )
    )


def _ensure_prereqs() -> None:
    missing = []
    if not DEEPGRAM_API_KEY:
        missing.append("DEEPGRAM_API_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not PRIVATE_LAYER_API_KEY:
        missing.append("PRIVATE_LAYER_API_KEY")
    if missing:
        print("Error: missing required environment variables:", ", ".join(missing))
        sys.exit(1)


def _init_log() -> Dict[str, Any]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    session_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_path = LOGS_DIR / f"session_{session_ts}.log"
    log_file = log_path.open("a", encoding="utf-8")
    return {"file": log_file, "path": log_path}


def _log_event_factory(handle):
    log_file = handle["file"]

    def _log(message: str) -> None:
        timestamp = datetime.utcnow().isoformat() + "Z"
        log_file.write(f"[{timestamp}] {message}\n")
        log_file.flush()

    return _log


def main() -> None:
    _ensure_prereqs()

    log_handle = _init_log()
    log_event = _log_event_factory(log_handle)
    print(f"[Log] Writing transcript + LLM payloads to {log_handle['path']}")

    greeting = get_greeting()
    log_event(
        f"Session started (stt_model={DEEPGRAM_LISTEN_MODEL} speak_model={DEEPGRAM_SPEAK_MODEL} openai_model={OPENAI_MODEL})"
    )
    log_event(f"System prompt:\n{build_system_prompt()}")
    log_event(f"Greeting: {greeting}")

    private_layer = PrivateLayerClient(
        base_url=PRIVATE_LAYER_URL,
        api_key=PRIVATE_LAYER_API_KEY,
        tenant_id=PRIVATE_LAYER_TENANT,
        detect_path=PRIVATE_LAYER_DETECT_PATH,
    )

    conversation: List[Dict[str, str]] = []

    print("Launching insurance agent (Deepgram STT/TTS + Private Layer + OpenAI).")
    print("Speak after the greeting and pause to finish a turn. Say 'goodbye' to exit.\n")
    _speak(greeting, log_event)
    print("[Agent]:", greeting)

    try:
        while True:
            print("\n[Listening] Speak now (pause to finish)...")
            try:
                audio_bytes = record_until_pause()
            except RuntimeError as exc:
                log_event(f"AUDIO_ERROR: {exc}")
                print("[Audio Error]", exc)
                break
            if not audio_bytes:
                print("[Audio] No input captured. Try again.")
                continue

            try:
                user_text = transcribe_pcm(audio_bytes, sample_rate=INPUT_SAMPLE_RATE)
            except DeepgramSTTError as exc:
                log_event(f"STT_ERROR: {exc}")
                print("[STT Error]", exc)
                _speak("Sorry, I couldn't understand that. Please repeat.", log_event)
                continue

            if not user_text:
                _speak("I didn't catch that. Could you repeat?", log_event)
                continue

            print("[Customer]:", user_text)
            log_event(f"USER_RAW: {user_text}")

            if _should_exit(user_text):
                farewell = "Thank you for calling Alpha Insurance. Goodbye!"
                _speak(farewell, log_event)
                print("[Agent]:", farewell)
                break

            try:
                sanitized = private_layer.sanitize(user_text)
            except PrivateLayerError as exc:
                log_event(f"PRIVATE_LAYER_ERROR: {exc}")
                print("[Private Layer Error]", exc)
                _speak("I ran into a privacy error. Let's try again in a moment.", log_event)
                continue

            conversation.append({"role": "user", "content": sanitized.text_with_placeholders})
            log_event(f"USER_SANITIZED: {sanitized.text_with_placeholders}")
            log_event(f"PRIVATE_LAYER_BUNDLES: {json.dumps(sanitized.bundles)}")

            try:
                assistant_text = generate_response(conversation)
            except OpenAIUnavailable as exc:
                log_event(f"OPENAI_ERROR: {exc}")
                print("[OpenAI Error]", exc)
                break
            except Exception as exc:  # pragma: no cover - propagate LLM issues
                log_event(f"OPENAI_ERROR: {exc}")
                print("[OpenAI Error]", exc)
                _speak("I'm having trouble with my brain right now. Let's try again shortly.", log_event)
                conversation.pop()
                continue

            log_event(f"LLM_OUTPUT: {assistant_text}")
            print("[Agent]:", assistant_text)

            for line in assistant_text.splitlines():
                update = _parse_policy_line(line)
                if update:
                    _log_policy(update, log_event)

            conversation.append({"role": "assistant", "content": assistant_text})
            _speak(assistant_text, log_event)

    except KeyboardInterrupt:
        print("\nStopping agent...")
        log_event("KeyboardInterrupt received. Shutting down agent loop.")
    finally:
        log_event("Session finished.")
        handle = log_handle.get("file")
        if handle:
            handle.close()

    print("Done.")


if __name__ == "__main__":
    main()
