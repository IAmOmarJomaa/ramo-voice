# 🎙️ ramO-Voice: Sovereign TTS & Voice Cloning Service

The standalone **ElevenLabs alternative** for the ramO Engine.

---

## 🚀 How to Run This Microservice Alone

```bash
# 1. Navigate to this service directory
cd services/tts

# 2. Run with UV or standard Python
uv run uvicorn ramo_voice.server:app --host 0.0.0.0 --port 50055 --reload
```

---

## 📡 Endpoints & Usage

### 1. Health Check
```bash
curl http://localhost:50055/health
```

### 2. List Available Voices (Presets & Cloned)
```bash
curl http://localhost:50055/v1/voices
```

### 3. Speech Synthesis (OpenAI & ElevenLabs Compatible)
```bash
curl -X POST http://localhost:50055/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Welcome to the sovereign voice engine. Studio audio at forty-four point one kilohertz.",
    "voice": "af_heart",
    "speed": 1.0
  }' \
  --output speech.wav
```

### 4. Zero-Shot Voice Cloning (Upload 3-10s WAV)
```bash
curl -X POST http://localhost:50055/v1/voices/clone \
  -F "file=@my_3s_sample.wav" \
  -F "voice_id=custom_omar" \
  -F "name=Omar Studio"
```

### 5. Real-Time Streaming WebSocket
Connect to `ws://localhost:50055/v1/stream`:
```json
{
  "text": "Streaming speech chunk by chunk with sub-100ms latency.",
  "voice": "af_heart"
}
```
The server yields raw 16-bit PCM binary chunks with 50ms equal-power crossfades.

---

## 🧪 Running Unit & Integration Tests

```bash
uv run pytest tests/ -v
```
