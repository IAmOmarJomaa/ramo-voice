# 🚪 ramO-Gateway: Sovereign Real-Time Voice Duplex Gateway

The standalone **Real-Time Voice Duplex Gateway & Barge-in Orchestrator** for the ramO Engine.

Coordinates full-duplex conversations with a turn-taking state machine:
- `LISTENING` -> `SPEAKING` -> `THINKING` -> `PLAYING`.
- **Zero-Latency Barge-In**: Instantly cuts off outgoing TTS audio when the user begins speaking.

---

## 🚀 How to Run This Microservice Alone

```bash
# 1. Navigate to this service directory
cd services/gateway

# 2. Run with UV or standard Python
uv run uvicorn ramo_gateway.server:app --host 0.0.0.0 --port 50050 --reload
```

---

## 📡 Endpoints & Usage

### 1. Health Check
```bash
curl http://localhost:50050/health
```

### 2. Duplex Conversational WebSocket
Connect to `ws://localhost:50050/v1/realtime`:
- Stream microphone audio bytes in.
- Receive synthesized TTS chunks back.
- If speech is detected while playback is occurring, the gateway automatically emits:
```json
{"type": "interrupt"}
```
allowing the client to immediately mute playback without audio overlap.

---

## 🧪 Running Unit & Integration Tests

```bash
uv run pytest tests/ -v
```
