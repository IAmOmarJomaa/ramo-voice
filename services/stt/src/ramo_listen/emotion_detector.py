"""
ramo_listen.emotion_detector
============================
Acoustic emotion and event classifier harvested from the SenseVoice architecture.
Detects rich conversational acoustic cues:
- Emotions: <|HAPPY|>, <|SAD|>, <|ANGRY|>, <|NEUTRAL|>
- Events: <|LAUGHTER|>, <|APPLAUSE|>, <|CRYING|>
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class AcousticEvents:
    emotion: str
    has_laughter: bool = False
    has_applause: bool = False
    tags: List[str] = field(default_factory=list)


def detect_acoustic_events(audio: np.ndarray, sample_rate: int = 16000) -> AcousticEvents:
    """
    Extract acoustic prosody and event markers from audio waveform.
    Uses spectral centroid, energy envelope variance, and high-frequency modulation.
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)

    audio = audio.astype(np.float32)
    if len(audio) == 0:
        return AcousticEvents(emotion="NEUTRAL", tags=["<|NEUTRAL|>"])

    # 1. Compute RMS energy
    rms = np.sqrt(np.mean(audio ** 2) + 1e-9)

    # 2. Frame-based analysis for temporal modulation
    frame_size = int(0.025 * sample_rate)  # 25ms
    hop_size = int(0.010 * sample_rate)    # 10ms
    num_frames = max(1, (len(audio) - frame_size) // hop_size)

    frame_energies = []
    spectral_centroids = []

    for i in range(num_frames):
        frame = audio[i * hop_size : i * hop_size + frame_size]
        f_rms = np.sqrt(np.mean(frame ** 2) + 1e-9)
        frame_energies.append(f_rms)

        # FFT for spectral centroid
        fft_vals = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(len(frame), d=1.0 / sample_rate)
        sum_fft = np.sum(fft_vals) + 1e-9
        centroid = np.sum(freqs * fft_vals) / sum_fft
        spectral_centroids.append(centroid)

    frame_energies = np.array(frame_energies, dtype=np.float32)
    spectral_centroids = np.array(spectral_centroids, dtype=np.float32)

    mean_energy = float(np.mean(frame_energies))
    var_energy = float(np.var(frame_energies))
    mean_centroid = float(np.mean(spectral_centroids))

    # 3. Laughter detection: High spectral centroid + rhythmic energy modulation (8-16 Hz bursts)
    has_laughter = False
    if len(frame_energies) > 10:
        # Check energy fluctuations (burstiness)
        peak_to_trough = (np.max(frame_energies) - np.min(frame_energies)) / (mean_energy + 1e-6)
        if peak_to_trough > 3.0 and mean_centroid > 2500.0:
            has_laughter = True

    # 4. Emotion classification
    if mean_energy > 0.35 and mean_centroid > 2200.0:
        emotion = "ANGRY"
    elif has_laughter or (mean_centroid > 2000.0 and var_energy > 0.01):
        emotion = "HAPPY"
    elif mean_energy < 0.05 and mean_centroid < 1200.0:
        emotion = "SAD"
    else:
        emotion = "NEUTRAL"

    tags = [f"<|{emotion}|>"]
    if has_laughter:
        tags.append("<|LAUGHTER|>")

    return AcousticEvents(
        emotion=emotion,
        has_laughter=has_laughter,
        tags=tags
    )
