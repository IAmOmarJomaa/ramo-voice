# 🎙️ ramO Engine: Sovereign Audio Intelligence Ecosystem

A production-grade, modular, self-hostable audio intelligence platform.
Built on a clean white paper as a suite of **independent sovereign microservices** that can be run alone or orchestrated together.

---

## 🏛️ Sovereign Microservices Topology

```text
C:\ramo-engine\
├── services/
│   ├── tts/              # [Microservice 1] ramO-Voice: The ElevenLabs Alternative
│   │   ├── Port: 50055   # OpenAI-compatible /v1/audio/speech, /v1/voices/clone & WebSocket
│   │   ├── Engines:      # Supertonic ONNX (Fast-Path) + Flow Matching (Zero-Shot Cloner)
│   │   └── Status:       # READY & TESTED (18/18 Tests Passing)
│   │
│   ├── stt/              # [Microservice 2] ramO-Listen: Real-time Streaming STT
│   │   ├── Port: 50051   # Whisper LocalAgreement (n=2) + SenseVoice emotion & acoustic tags
│   │   └── Status:       # READY & TESTED (10/10 Tests Passing)
│   │
│   ├── diarization/      # [Microservice 3] ramO-Speaker: Speaker Diarization & Harvester
│   │   ├── Port: 50052   # Pyannote clustering + StenoAI crosstalk rejection + Voiceprint harvester
│   │   └── Status:       # READY & TESTED (7/7 Tests Passing)
│   │
│   └── gateway/          # [Microservice 4] ramO-Gateway: Real-Time Duplex Orchestrator
│       ├── Port: 50050   # Turn-taking VAD state machine & zero-latency barge-in
│       └── Status:       # READY & TESTED (4/4 Tests Passing)
└── deploy/               # Colab T4 Deployment Suite (15GB VRAM allocation budget)
    ├── colab_run.sh
    └── README.md
```

---

## 🧪 Comprehensive Verification Summary

All 4 microservices have zero cross-dependencies and maintain their own test suites:

| Microservice | Test Path | Tests Passed | Status |
| :--- | :--- | :--- | :--- |
| **`services/tts`** | `services/tts/tests/` | **18 / 18** | ✅ GREEN |
| **`services/stt`** | `services/stt/tests/` | **10 / 10** | ✅ GREEN |
| **`services/diarization`** | `services/diarization/tests/` | **7 / 7** | ✅ GREEN |
| **`services/gateway`** | `services/gateway/tests/` | **4 / 4** | ✅ GREEN |
| **TOTAL** | | **39 / 39** | **100% GREEN** |

---

## 🚀 Running Any Service Standalone

### 1. TTS & Voice Cloning (`services/tts`)
```bash
cd services/tts
uv run uvicorn ramo_voice.server:app --host 0.0.0.0 --port 50055 --reload
# Tests: uv run pytest tests/ -v
```

### 2. Streaming STT & Emotions (`services/stt`)
```bash
cd services/stt
uv run uvicorn ramo_listen.server:app --host 0.0.0.0 --port 50051 --reload
# Tests: uv run pytest tests/ -v
```

### 3. Diarization & Voiceprint Harvester (`services/diarization`)
```bash
cd services/diarization
uv run uvicorn ramo_speaker.server:app --host 0.0.0.0 --port 50052 --reload
# Tests: uv run pytest tests/ -v
```

### 4. Duplex Real-Time Gateway (`services/gateway`)
```bash
cd services/gateway
uv run uvicorn ramo_gateway.server:app --host 0.0.0.0 --port 50050 --reload
# Tests: uv run pytest tests/ -v
```

### 5. Colab T4 Cloud Launch
```bash
bash deploy/colab_run.sh
```
