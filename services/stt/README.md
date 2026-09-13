# 👂 ramO-Listen: Sovereign Streaming STT & Emotion Microservice

The standalone **Real-Time Speech-to-Text & Acoustic Event Engine** for the ramO Engine.

Combines the **LocalAgreement $n=2$** streaming algorithm from `whisper_streaming` with **SenseVoice** sub-50ms acoustic emotion and event intelligence.

---

## 🚀 How to Run This Microservice Alone

```bash
# 1. Navigate to this service directory
cd services/stt

# 2. Run with UV or standard Python
uv run uvicorn ramo_listen.server:app --host 0.0.0.0 --port 50051 --reload
```

---

## 📡 Endpoints & Usage

### 1. Health Check
```bash
curl http://localhost:50051/health
```

### 2. Audio Transcription (OpenAI Compatible + SenseVoice Tags)
```bash
curl -X POST http://localhost:50051/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "language=en"
```
**Response JSON:**
```json
{
  "text": "<|HAPPY|> <|LAUGHTER|> Welcome to the show everyone",
  "raw_text": "Welcome to the show everyone",
  "emotion": "HAPPY",
  "has_laughter": true,
  "tags": ["<|HAPPY|>", "<|LAUGHTER|>"],
  "duration": 2.45,
  "confidence": 0.98
}
```

### 3. Real-Time Streaming WebSocket
Connect to `ws://localhost:50051/v1/listen`:
- Stream raw 16kHz 16-bit PCM bytes continuously into the WebSocket.
- The server responds in real time with stabilized partials:
```json
{
  "type": "partial",
  "committed": "welcome to the",
  "tentative": "stream",
  "emotion": "NEUTRAL",
  "tags": ["<|NEUTRAL|>"]
}
```
- Send `{"type": "flush"}` to finalize the sentence upon pause/silence.

---

## 🏛️ Core Architectural Invariants

1. **LocalAgreement ($n=2$)**:
   - Compares consecutive overlapping predictions. Words are only committed when agreed upon by 2 successive windows.
   - Prevents the runaway word deletion and hallucination cascades inherent in naive sliding-window Whisper.

2. **Acoustic Event & Emotion Classification**:
   - Frame-level spectral centroid and energy modulation analysis.
   - Tags conversational dynamics (`<|HAPPY|>`, `<|ANGRY|>`, `<|SAD|>`, `<|LAUGHTER|>`).

3. **Zero-Latency Audio Ring Buffer**:
   - Handles continuous audio ingestion with automatic sample-rate resampling from 44.1kHz / 48kHz down to 16kHz.

---

## 🧪 Running Unit & Integration Tests

```bash
uv run pytest tests/ -v
```
