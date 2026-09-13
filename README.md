# 🎙️ ramO Engine: Sovereign Audio Intelligence Ecosystem

A production-grade, modular, self-hostable audio intelligence platform.
Built as a suite of **independent sovereign microservices** that can be run alone or orchestrated together.

---

## 🏛️ Sovereign Microservices Topology

```text
C:\ramo-engine\
├── services/
│   ├── tts/              # [Microservice 1] ramO-Voice: The ElevenLabs Alternative
│   │   ├── Port: 50055   # OpenAI-compatible /v1/audio/speech & real-time WebSocket
│   │   └── Status: READY & TESTED (14/14 Unit Tests Passing)
│   │
│   ├── stt/              # [Microservice 2] Real-time Streaming ASR
│   │   ├── Port: 50051   # LocalAgreement (n=2) Whisper + SenseVoice emotion tags
│   │   └── Status: IN DESIGN
│   │
│   ├── diarization/      # [Microservice 3] Speaker Tracking & Profiling
│   │   ├── Port: 50052   # Pyannote community-1 + Demucs vocal cleaner + Centroids
│   │   └── Status: IN DESIGN
│   │
│   └── gateway/          # [Microservice 4] Unified Client Ingress
│       ├── Port: 8000    # Client WebSocket router & audio multiplexer
│       └── Status: IN DESIGN
```

---

## 🚀 Quickstart: Running Services Standalone

Each service is 100% sovereign. You can boot any service independently without needing the others:

### 1. Run the ElevenLabs TTS Engine Standalone:
```bash
cd services/tts
uv run uvicorn ramo_voice.server:app --host 0.0.0.0 --port 50055 --reload
```
- Test health: `curl http://localhost:50055/health`
- List voices: `curl http://localhost:50055/v1/voices`
- Generate audio: `curl -X POST http://localhost:50055/v1/audio/speech -H "Content-Type: application/json" -d "{\"input\": \"Hello from clean ramO engine!\"}" --output speech.wav`
