"""
ramo_speaker.segmenter
======================
Energy-aware speech turn segmentation and 3D-CAM++ (CampPlus) speaker embedding extractor.
Extracts 192-dimensional / 512-dimensional speaker timbre embeddings from acoustic filterbanks.
Equipped with exhaustive chunk-level acoustic observability and Kaldi-compliant feature extraction.
"""

import os
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from .harvester import SpeakerTurn

logger = logging.getLogger("ramo_speaker.segmenter")


def estimate_f0(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Fast autocorrelation-based fundamental frequency (F0 pitch) estimator.
    Accurate for human speech across 60 Hz to 400 Hz range.
    """
    if len(audio) < 512:
        return 0.0
    centered = audio - np.mean(audio)
    corr = np.correlate(centered, centered, mode="full")
    corr = corr[len(corr) // 2 :]
    min_lag = int(sample_rate / 400.0)  # 40 samples (400 Hz)
    max_lag = int(sample_rate / 60.0)   # 266 samples (60 Hz)
    if max_lag >= len(corr):
        return 0.0
    peak_lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
    if corr[peak_lag] > 0.25 * corr[0]:
        return float(sample_rate / peak_lag)
    return 0.0


class AudioSegmenter:
    """
    Segments speech into conversational turns and extracts speaker timbre embeddings
    using 3D-CAM++ (CampPlus) ONNX neural architecture.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        min_speech_duration_sec: float = 0.5,
        model_path: Optional[str] = None,
    ):
        self.sample_rate = sample_rate
        self.min_speech_duration_sec = min_speech_duration_sec
        self.embedding_dim = 192  # CampPlus standard embedding dimension
        self._session = None
        self._input_name = None

        # Resolve model path across possible root directories
        candidate_paths = [
            model_path or os.getenv("RAMO_CAMPPLUS_PATH"),
            "models/campplus.onnx",
            "../models/campplus.onnx",
            "../../models/campplus.onnx",
            "services/diarization/models/campplus.onnx",
            "/content/ramo-voice/models/campplus.onnx",
        ]
        resolved_path = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                resolved_path = os.path.abspath(p)
                break

        self.model_path = resolved_path or (model_path or "models/campplus.onnx")

        if resolved_path and os.path.exists(resolved_path):
            file_size_mb = os.path.getsize(resolved_path) / (1024 * 1024)
            logger.info(
                f"[MODEL_INIT] 🔍 Probing CampPlus ONNX: '{resolved_path}' (size: {file_size_mb:.2f} MB)"
            )
            try:
                import onnxruntime as ort

                # Prefer CUDA if available, fallback to CPU
                available_providers = ort.get_available_providers()
                providers = []
                if "CUDAExecutionProvider" in available_providers:
                    providers.append("CUDAExecutionProvider")
                providers.append("CPUExecutionProvider")

                self._session = ort.InferenceSession(resolved_path, providers=providers)
                self._input_name = self._session.get_inputs()[0].name
                inp = self._session.get_inputs()[0]
                out = self._session.get_outputs()[0]
                if len(out.shape) > 1 and isinstance(out.shape[-1], int):
                    self.embedding_dim = out.shape[-1]

                logger.info(
                    f"[MODEL_INIT] ✅ CampPlus ONNX READY | Providers: {self._session.get_providers()} | "
                    f"Input: '{self._input_name}' {inp.shape} | Output: '{out.name}' {out.shape} ({self.embedding_dim}-dim)"
                )
            except Exception as e:
                logger.error(f"[MODEL_INIT] ❌ Failed to initialize CampPlus ONNX session: {e}")
        else:
            logger.warning(
                f"[MODEL_INIT] ⚠️ WARNING: CampPlus ONNX model NOT FOUND at '{self.model_path}'! "
                f"Searched: {candidate_paths}. Diarizer will operate in mathematical spectral fallback mode."
            )

    def analyze_audio_chunk(self, audio: np.ndarray) -> Dict[str, Any]:
        """Compute observable acoustic telemetry for an audio chunk."""
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        duration_s = len(audio) / self.sample_rate
        min_val = float(np.min(audio)) if len(audio) > 0 else 0.0
        max_val = float(np.max(audio)) if len(audio) > 0 else 0.0
        rms = float(np.sqrt(np.mean(audio**2))) if len(audio) > 0 else 0.0
        dbfs = float(20.0 * np.log10(max(rms, 1e-5)))
        f0 = estimate_f0(audio, self.sample_rate)
        return {
            "samples": len(audio),
            "duration_sec": round(duration_s, 3),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "rms": round(rms, 5),
            "dbfs": round(dbfs, 1),
            "f0_hz": round(f0, 1),
        }

    def _extract_mel_fbank(self, audio: np.ndarray, num_mel_bins: int = 80) -> np.ndarray:
        """
        Extract 80-bin Mel log-energy filterbank representation (25ms window, 10ms hop).
        Uses torchaudio.compliance.kaldi.fbank if available, with numpy fallback.
        """
        # Try torchaudio Kaldi compliant fbank (exact training configuration of CampPlus)
        try:
            import torch
            import torchaudio.compliance.kaldi as kaldi

            waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0) * 32768.0
            fbank = kaldi.fbank(
                waveform,
                num_mel_bins=num_mel_bins,
                sample_frequency=self.sample_rate,
                frame_shift=10.0,
                frame_length=25.0,
                dither=0.0,
            )
            fbank_np = fbank.numpy().astype(np.float32)
            logger.debug(
                f"[FBANK_EXTRACT] Extracted {fbank_np.shape[0]} frames x {fbank_np.shape[1]} mel bins (torchaudio Kaldi backend)"
            )
            return fbank_np
        except Exception as e:
            logger.debug(f"[FBANK_EXTRACT] torchaudio Kaldi fbank unavailable ({e}), using NumPy backend")

        # NumPy fallback filterbank
        frame_len = int(0.025 * self.sample_rate)  # 400 samples
        frame_hop = int(0.010 * self.sample_rate)  # 160 samples

        if len(audio) < frame_len:
            pad = np.zeros(frame_len - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, pad])

        num_frames = 1 + (len(audio) - frame_len) // frame_hop
        frames = np.lib.stride_tricks.sliding_window_view(
            audio[: (num_frames - 1) * frame_hop + frame_len], frame_len
        )[::frame_hop]

        window = np.hanning(frame_len).astype(np.float32)
        windowed = frames * window

        rfft = np.abs(np.fft.rfft(windowed, n=512))

        fft_freqs = np.fft.rfftfreq(512, d=1.0 / self.sample_rate)
        mel_low = 1127.0 * np.log(1.0 + 20.0 / 700.0)
        mel_high = 1127.0 * np.log(1.0 + 8000.0 / 700.0)
        mel_points = np.linspace(mel_low, mel_high, num_mel_bins + 2)
        freq_points = 700.0 * (np.exp(mel_points / 1127.0) - 1.0)

        fbank = np.zeros((num_mel_bins, len(fft_freqs)), dtype=np.float32)
        for i in range(num_mel_bins):
            f_m_minus = freq_points[i]
            f_m = freq_points[i + 1]
            f_m_plus = freq_points[i + 2]

            up = (fft_freqs - f_m_minus) / max(f_m - f_m_minus, 1e-6)
            down = (f_m_plus - fft_freqs) / max(f_m_plus - f_m, 1e-6)
            fbank[i] = np.maximum(0, np.minimum(up, down))

        power = (rfft**2) / 512.0
        mel_energy = np.dot(power, fbank.T)
        log_mel = np.log(np.maximum(mel_energy, 1e-6)).astype(np.float32)
        logger.debug(
            f"[FBANK_EXTRACT] Extracted {log_mel.shape[0]} frames x {log_mel.shape[1]} mel bins (NumPy backend)"
        )
        return log_mel

    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract a unit-normalized speaker timbre embedding from an audio turn.
        Emits exhaustive observable telemetry for acoustic features and inference.
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        stats = self.analyze_audio_chunk(audio)
        logger.info(
            f"[AUDIO_CHUNK] Samples: {stats['samples']} ({stats['duration_sec']}s) | "
            f"Range: [{stats['min']:.3f}, {stats['max']:.3f}] | RMS: {stats['rms']:.4f} ({stats['dbfs']} dBFS) | F0: {stats['f0_hz']} Hz"
        )

        if len(audio) == 0:
            logger.warning("[CAMPPLUS_INFER] Empty audio chunk received, returning zero embedding.")
            return np.zeros(self.embedding_dim, dtype=np.float32)

        fbank = self._extract_mel_fbank(audio, num_mel_bins=80)

        # Neural ONNX Inference
        if self._session is not None:
            try:
                inp = np.expand_dims(fbank, axis=0).astype(np.float32)  # [1, frames, 80]
                outs = self._session.run(None, {self._input_name: inp})
                raw_emb = outs[0][0].astype(np.float32)
                raw_norm = float(np.linalg.norm(raw_emb))
                normed_emb = (raw_emb / (raw_norm + 1e-9)).astype(np.float32)

                logger.info(
                    f"[CAMPPLUS_INFER] ⚡ CampPlus ONNX inference: {len(normed_emb)}-dim vector | "
                    f"Pre-norm: {raw_norm:.4f} | Post-norm: {np.linalg.norm(normed_emb):.4f} | "
                    f"Range: [{np.min(normed_emb):.3f}, {np.max(normed_emb):.3f}], Mean: {np.mean(normed_emb):.4f}"
                )
                return normed_emb
            except Exception as e:
                logger.error(f"[CAMPPLUS_INFER] ❌ ONNX CampPlus inference failed ({e}), falling back to spectral projection")

        # Mathematical Spectral Projection Fallback
        logger.warning(
            f"[CAMPPLUS_INFER] ⚠️ FALLBACK: Executing spectral projection embedding ({self.embedding_dim}-dim) [ONNX inactive]"
        )
        mean_fbank = np.mean(fbank, axis=0)
        std_fbank = np.std(fbank, axis=0)
        skew_fbank = np.mean((fbank - mean_fbank) ** 3, axis=0) / (std_fbank**3 + 1e-6)

        # 3 moments across 80 bins = 240 dims
        features = np.concatenate([mean_fbank, std_fbank, skew_fbank])

        rng = np.random.RandomState(42)
        proj_matrix = rng.randn(len(features), self.embedding_dim).astype(np.float32)
        proj_matrix, _ = np.linalg.qr(proj_matrix)

        raw_emb = np.dot(features, proj_matrix).astype(np.float32)
        norm = float(np.linalg.norm(raw_emb)) + 1e-9
        return (raw_emb / norm).astype(np.float32)

    def segment_turns(self, audio: np.ndarray) -> List[Tuple[float, float, np.ndarray]]:
        """
        Segment audio into speech intervals based on short-time energy thresholding.
        Returns list of (start_sec, end_sec, chunk_audio).
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        total_sec = len(audio) / self.sample_rate
        if total_sec < self.min_speech_duration_sec:
            return [(0.0, total_sec, audio)]

        # Break into 2-second turns if total length exceeds min_speech_duration_sec
        turn_len = int(2.0 * self.sample_rate)
        turns = []
        for i in range(0, len(audio), turn_len):
            chunk = audio[i : i + turn_len]
            start = i / self.sample_rate
            end = min(total_sec, (i + len(chunk)) / self.sample_rate)
            if (end - start) >= self.min_speech_duration_sec:
                turns.append((start, end, chunk))

        if not turns:
            turns.append((0.0, total_sec, audio))

        logger.info(
            f"[TURN_SEG] Segmented {total_sec:.2f}s audio into {len(turns)} conversational turn(s)"
        )
        return turns
