"""Load runtime configuration for the Deepgram insurance agent."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
POLICY_FILE = DATA_DIR / "policy_updates.json"
LOGS_DIR = DATA_DIR / "logs"

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_LISTEN_MODEL = os.getenv("DEEPGRAM_LISTEN_MODEL", "nova-3")
DEEPGRAM_SPEAK_MODEL = os.getenv("DEEPGRAM_SPEAK_MODEL", "aura-2-thalia-en")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

PRIVATE_LAYER_URL = os.getenv("PRIVATE_LAYER_URL", "http://127.0.0.1:9000")
PRIVATE_LAYER_API_KEY = os.getenv("PRIVATE_LAYER_API_KEY", "dev-secret-demo")
PRIVATE_LAYER_TENANT = os.getenv("PRIVATE_LAYER_TENANT", "default")
PRIVATE_LAYER_DETECT_PATH = os.getenv("PRIVATE_LAYER_DETECT_PATH", "/v1/detect-encrypt")
PRIVATE_LAYER_DECRYPT_PATH = os.getenv("PRIVATE_LAYER_DECRYPT_PATH", "/v1/decrypt")
try:
    PRIVATE_LAYER_TIMEOUT = float(os.getenv("PRIVATE_LAYER_TIMEOUT", "60"))
except ValueError:
    PRIVATE_LAYER_TIMEOUT = 60.0

# Deepgram REST defaults
DEEPGRAM_TTS_SAMPLE_RATE = 24000
DEEPGRAM_TTS_ENCODING = "linear16"
DEEPGRAM_STT_ENDPOINT = os.getenv("DEEPGRAM_STT_ENDPOINT", "https://api.deepgram.com/v1/listen")
DEEPGRAM_TTS_ENDPOINT = os.getenv("DEEPGRAM_TTS_ENDPOINT", "https://api.deepgram.com/v1/speak")

# Default audio params recommended by Deepgram docs
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHUNK_MS = 100
CHUNK_SAMPLES = int(INPUT_SAMPLE_RATE * CHUNK_MS / 1000)
