"""
ramo_voice.config
=================
Central configuration for the sovereign voice engine.
"""

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
VOICES_DIR = BASE_DIR / "voice_profiles"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# Performance & Streaming Gates
DEFAULT_MAX_CHUNK_CHARS = 400
DEFAULT_CROSSFADE_MS = 50
DEFAULT_SAMPLE_RATE = 44100

# Server Defaults
HOST = os.environ.get("RAMO_VOICE_HOST", "0.0.0.0")
PORT = int(os.environ.get("RAMO_VOICE_PORT", "50055"))
DEVICE = os.environ.get("RAMO_VOICE_DEVICE", "cpu")  # cpu or cuda
