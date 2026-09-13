"""
ramo_voice.audio_utils
======================
Audio pre/post-processing utilities: runaway detection, silence trimming, normalization.
Extracted from voicebox's audio intelligence module.
"""

import numpy as np


def has_tts_runaway(
    audio: np.ndarray,
    sample_rate: int = 24000,
    frame_ms: int = 20,
    silence_threshold_db: float = -40.0,
    max_internal_silence_ms: int = 2000,
) -> bool:
    """
    Detect speech followed by a long silence (>2s) and resumed audio.
    This pattern reliably indicates that a TTS model missed the EOS token
    and resumed hallucinating breathing or noise.
    """
    frame_len = int(sample_rate * frame_ms / 1000)
    if frame_len == 0 or len(audio) < frame_len:
        return False

    n_frames = len(audio) // frame_len
    threshold_linear = 10 ** (silence_threshold_db / 20)
    max_silence_frames = int(max_internal_silence_ms / frame_ms)
    seen_speech = False
    consecutive_silence = 0

    for i in range(n_frames):
        frame = audio[i * frame_len:(i + 1) * frame_len]
        is_speech = np.sqrt(np.mean(frame ** 2)) >= threshold_linear
        if is_speech:
            if seen_speech and consecutive_silence >= max_silence_frames:
                return True
            seen_speech = True
            consecutive_silence = 0
        elif seen_speech:
            consecutive_silence += 1

    return False


def trim_tts_output(
    audio: np.ndarray,
    sample_rate: int = 24000,
    frame_ms: int = 20,
    silence_threshold_db: float = -40.0,
    min_silence_ms: int = 200,
    max_internal_silence_ms: int = 1000,
    fade_ms: int = 30,
) -> np.ndarray:
    """
    Trim trailing silence and post-silence hallucination from TTS output.
    Applies a smooth cosine fade-out.
    """
    frame_len = int(sample_rate * frame_ms / 1000)
    if frame_len == 0 or len(audio) < frame_len:
        return audio

    n_frames = len(audio) // frame_len
    threshold_linear = 10 ** (silence_threshold_db / 20)
    max_silence_frames = int(max_internal_silence_ms / frame_ms)
    min_silence_frames = int(min_silence_ms / frame_ms)

    # 1. Check for internal runaway silence gap
    seen_speech = False
    silence_start = -1
    cut_frame = -1

    for i in range(n_frames):
        frame = audio[i * frame_len:(i + 1) * frame_len]
        is_speech = np.sqrt(np.mean(frame ** 2)) >= threshold_linear
        if is_speech:
            seen_speech = True
            silence_start = -1
        elif seen_speech:
            if silence_start == -1:
                silence_start = i
            elif (i - silence_start) >= max_silence_frames:
                cut_frame = silence_start + min_silence_frames
                break

    if cut_frame > 0:
        audio = audio[:cut_frame * frame_len]

    # 2. Trim trailing silence and apply cosine fade
    n_frames = len(audio) // frame_len
    last_speech_frame = 0
    for i in range(n_frames - 1, -1, -1):
        frame = audio[i * frame_len:(i + 1) * frame_len]
        if np.sqrt(np.mean(frame ** 2)) >= threshold_linear:
            last_speech_frame = i
            break

    end_sample = min(len(audio), (last_speech_frame + min_silence_frames + 1) * frame_len)
    audio = audio[:end_sample]

    # 3. Cosine fade out
    fade_samples = int(sample_rate * fade_ms / 1000)
    if len(audio) > fade_samples:
        fade_curve = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_samples)))
        audio[-fade_samples:] *= fade_curve

    return audio


def normalize_audio(audio: np.ndarray, target_db: float = -24.0) -> np.ndarray:
    """Peak-normalize audio array to target dBFS."""
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio
    target_linear = 10 ** (target_db / 20)
    scalar = target_linear / max_val
    return (audio * scalar).astype(np.float32)
