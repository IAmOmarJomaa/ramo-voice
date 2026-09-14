# ramO-Clean: Sovereign 5-Stage Audio Preconditioning & Intelligence

`services/enhancement` is the sovereign audio preconditioning engine for the ramO ecosystem. It operates in two modes:
1. **In-Memory Zero-Latency Library**: Direct C/NumPy execution in $<1.2\text{ms}$ (`from ramo_clean import AudioPreconditioner`).
2. **Standalone REST & WebSocket Microservice**: FastAPI server on Port `50054` for independent testing and external agent integration.

---

## 5-Stage Preconditioning Architecture

```
Incoming Audio (44.1kHz / 48kHz / 16kHz)
  │
  ▼
Stage 1: 80Hz Butterworth High-Pass Filter (HPF)
  │      • 4th-order filter cutting HVAC rumble & desk thumps (< 80Hz)
  │      • Preserves 100% of human voice fundamentals (100Hz - 300Hz)
  ▼
Stage 2: Frequency-Domain Spectral Gating
  │      • STFT dynamic noise floor estimation
  │      • Attenuates stationary acoustic hiss and room noise
  ▼
Stage 3: Adaptive Gain Control (AGC) & Soft-Knee Limiter
  │      • Dynamic normalization to target -20 dBFS
  │      • Soft-knee compression preventing digital clipping (> 1.0)
  ▼
Stage 4: Silero VAD v5 ONNX Engine
  │      • 1.2MB ONNX model on CPU (0 MB GPU VRAM, < 0.8ms inference)
  │      • Dual-threshold hysteresis (0.50 / 0.35)
  │      • 300ms pre-pad (zero clipped first syllables)
  │      • 400ms post-pad (captures trailing breath / consonants)
  │      • 600ms redemption window (bridges natural breath pauses)
  ▼
Stage 5: Optional Demucs 2-Stem Vocal Separation
  │      • High-noise environment override for cafe/babble/music isolation
  ▼
Clean 16kHz Float32 PCM Out
```

---

## Direct In-Memory Library Usage

```python
from ramo_clean import AudioPreconditioner

preconditioner = AudioPreconditioner(sample_rate=16000)

# Process 1D float32 audio chunk in < 1.2ms
result = preconditioner.process_chunk(raw_chunk)
clean_audio = result.audio
is_speech = result.is_speech
vad_events = result.vad_events
```

---

## Standalone Server API (Port 50054)

### Start Server
```bash
uvicorn ramo_clean.server:app --port 50054
```

### Endpoints
- `GET /v1/health`: Health status & stage audit.
- `POST /v1/audio/clean`: Upload WAV file $\to$ receive cleaned 16-bit PCM WAV.
- `POST /v1/audio/vad`: Upload WAV file $\to$ receive speech state and boundary events.
- `WS /v1/audio/clean/stream`: Low-latency duplex binary PCM WebSocket stream.
