"""
ramo_clean
==========
Sovereign 5-Stage Audio Preconditioning & Intelligence Microservice.
Stages:
1. 80Hz Butterworth High-Pass Filter (HPF)
2. Frequency-domain Spectral Gating & Noise Suppression
3. Adaptive Gain Control (AGC) & Soft-Knee Limiter (-20 dBFS target)
4. Silero VAD v5 ONNX Engine (Hysteresis, Pre-pad, Post-pad, Redemption)
5. Optional Demucs 2-Stem Vocal Separation
"""

from ramo_clean.hpf import HPFFilter
from ramo_clean.agc import AGCLeveler
from ramo_clean.spectral_gate import SpectralGate
from ramo_clean.silero_vad import SileroProcessor, SpeechStart, SpeechEnd
from ramo_clean.pipeline import AudioPreconditioner, PreconditionedAudio

__all__ = [
    "HPFFilter",
    "AGCLeveler",
    "SpectralGate",
    "SileroProcessor",
    "SpeechStart",
    "SpeechEnd",
    "AudioPreconditioner",
    "PreconditionedAudio",
]
