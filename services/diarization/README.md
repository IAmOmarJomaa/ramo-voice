# 👥 ramO-Diarization: Sovereign Speaker Diarization & Voiceprint Harvester

The standalone **Speaker Diarization & Voiceprint Harvester Microservice** for the ramO Engine.

Combines **Pyannote-style** cosine distance clustering with **StenoAI's crosstalk rejection** and an automated **Voiceprint Harvester** to isolate clean, non-overlapping speaker samples for immediate zero-shot TTS cloning.

---

## 🚀 How to Run This Microservice Alone

```bash
# 1. Navigate to this service directory
cd services/diarization

# 2. Run with UV or standard Python
uv run uvicorn ramo_speaker.server:app --host 0.0.0.0 --port 50052 --reload
```

---

## 📡 Endpoints & Usage

### 1. Health Check
```bash
curl http://localhost:50052/health
```

### 2. Diarize Conversation
```bash
curl -X POST http://localhost:50052/v1/diarize \
  -F "file=@meeting_recording.wav"
```
**Response JSON:**
```json
{
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "turns": [
    {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "duration": 2.0},
    {"speaker": "SPEAKER_01", "start": 2.0, "end": 4.5, "duration": 2.5}
  ],
  "total_turns": 2
}
```

### 3. Voiceprint Harvesting (Zero-Shot Cloning Prep)
```bash
curl -X POST http://localhost:50052/v1/harvest/voiceprint \
  -F "file=@meeting_recording.wav"
```
**Response JSON:**
```json
{
  "status": "success",
  "harvested_speakers": [
    {
      "speaker_id": "SPEAKER_01",
      "duration_sec": 2.5,
      "samples_count": 40000,
      "ready_for_cloning": true
    }
  ],
  "total_harvested": 1
}
```

---

## 🏛️ Core Architectural Invariants

1. **Crosstalk & Overlap Rejection**:
   - Overlapping speech segments corrupt speaker centroids. Any turn flagged as overlap is excluded from centroid updates and voiceprint harvesting.

2. **Rolling Centroid Momentum**:
   - Updates speaker centroids with an exponential moving average ($c_t = \alpha c_{t-1} + (1-\alpha) e_t$) to track vocal fatigue and microphone distance without merging distinct speakers.

---

## 🧪 Running Unit & Integration Tests

```bash
uv run pytest tests/ -v
```
