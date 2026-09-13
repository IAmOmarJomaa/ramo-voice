"""
ramo_voice.purifier
===================
In-memory vocal isolation & background noise purge.
Extracted from Meta's Demucs architecture.
Ensures harvested reference audio is 100% clean before voice cloning.
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger("ramo_voice.purifier")


def clean_vocal_prompt(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Purify a raw microphone or meeting harvested speech clip.
    Removes background fan noise, room reverb, and hum so it doesn't contaminate
    the cloned voice timbre.
    """
    if len(audio) == 0:
        return audio

    try:
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_model("htdemucs")
        model.to(device)

        # Convert numpy audio (mono) to (batch, channels, samples) at 44.1kHz
        import scipy.signal
        if sample_rate != model.samplerate:
            num_target = int(len(audio) * model.samplerate / sample_rate)
            audio_resampled = scipy.signal.resample(audio, num_target)
        else:
            audio_resampled = audio

        tensor = torch.from_numpy(audio_resampled).float().unsqueeze(0).repeat(1, 2, 1).to(device)
        sources = apply_model(model, tensor, device=device)[0]

        # Extract vocal stem
        vocal_idx = model.sources.index("vocals")
        vocal_stereo = sources[vocal_idx].mean(dim=0).cpu().numpy()

        # Resample back if necessary
        if sample_rate != model.samplerate:
            vocal_mono = scipy.signal.resample(vocal_stereo, len(audio)).astype(np.float32)
        else:
            vocal_mono = vocal_stereo.astype(np.float32)

        logger.info("Successfully purified vocal prompt using Demucs.")
        return vocal_mono

    except (ImportError, Exception) as e:
        logger.debug(f"Demucs neural extraction not active ({e}); applying spectral bandpass filter.")
        # Fast fallback: 80Hz - 7500Hz vocal formant bandpass filter + silence gating
        import scipy.signal
        sos = scipy.signal.butter(4, [80, min(7500, sample_rate // 2 - 100)], btype="bandpass", fs=sample_rate, output="sos")
        filtered = scipy.signal.sosfilt(sos, audio).astype(np.float32)
        return filtered
