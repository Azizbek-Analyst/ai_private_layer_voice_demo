"""Microphone recording + playback helpers."""
from __future__ import annotations

import queue
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:  # pragma: no cover - environment detail
    SOUNDDEVICE_AVAILABLE = False

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:  # pragma: no cover
    WEBRTC_VAD_AVAILABLE = False

INPUT_SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(INPUT_SAMPLE_RATE * FRAME_MS / 1000)


def _record_fixed_duration(seconds: float) -> bytes:
    if not SOUNDDEVICE_AVAILABLE:
        raise RuntimeError("sounddevice not installed. pip install sounddevice")
    num_samples = int(INPUT_SAMPLE_RATE * seconds)
    recording = sd.rec(num_samples, samplerate=INPUT_SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    return recording.flatten().tobytes()


def record_until_pause(
    *,
    silence_seconds: float = 1.2,
    max_seconds: float = 30.0,
    fallback_seconds: float = 5.0,
) -> bytes:
    """Record from the microphone until silence or timeout. Returns PCM 16-bit mono bytes."""
    if not SOUNDDEVICE_AVAILABLE:
        raise RuntimeError("sounddevice not installed. pip install sounddevice")

    if not WEBRTC_VAD_AVAILABLE:
        # Fallback to fixed duration capture when VAD is missing
        return _record_fixed_duration(fallback_seconds)

    vad = webrtcvad.Vad(2)
    silence_frames_threshold = int(silence_seconds * 1000 / FRAME_MS)
    max_frames = int(max_seconds * 1000 / FRAME_MS)
    frames: queue.Queue[np.ndarray] = queue.Queue()

    state = {"silent_frames": 0, "started": False}

    def callback(indata, _frames, _time, status):
        if status:  # pragma: no cover - hardware dependent
            print("[Audio] status:", status)
        chunk = (indata[:, 0] * 32767).astype(np.int16)
        for idx in range(0, len(chunk), FRAME_SAMPLES):
            frame = chunk[idx : idx + FRAME_SAMPLES]
            if frame.shape[0] < FRAME_SAMPLES:
                break
            is_speech = vad.is_speech(frame.tobytes(), INPUT_SAMPLE_RATE)
            if is_speech:
                state["silent_frames"] = 0
                state["started"] = True
            elif state["started"]:
                state["silent_frames"] += 1
            frames.put(frame.copy())

    stream = sd.InputStream(
        samplerate=INPUT_SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        channels=1,
        dtype="float32",
        callback=callback,
    )
    stream.start()
    try:
        while frames.qsize() < max_frames:
            sd.sleep(int(FRAME_MS))
            if state["started"] and state["silent_frames"] >= silence_frames_threshold:
                break
        stream.stop()
    finally:
        stream.close()

    if frames.empty():
        return b""
    buffers = []
    while not frames.empty():
        buffers.append(frames.get())
    audio = np.concatenate(buffers)
    return audio.tobytes()


def play_pcm16(audio_bytes: bytes, sample_rate: int) -> None:
    """Play raw PCM16 mono audio."""
    if not SOUNDDEVICE_AVAILABLE:
        print("[Audio] sounddevice missing; printing text response only.")
        return
    if not audio_bytes:
        return
    arr = np.frombuffer(audio_bytes, dtype=np.int16)
    sd.play(arr, samplerate=sample_rate)
    sd.wait()
