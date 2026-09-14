#!/usr/bin/env bash
# ==============================================================================
# ramO Sovereign Engine: Colab T4 Cloud Deployment & Multi-Service Runner
# Strict VRAM Budgeting: ~7.8 GB Active / ~7.2 GB Free on 15GB T4 GPU
# Pure Python AsyncIO Orchestration (Zero Go Code)
# ==============================================================================

set -e

echo "======================================================================"
echo "🚀 Starting Sovereign ramO Engine Multi-Service Swarm on Colab T4"
echo "======================================================================"

# 1. GPU & VRAM Diagnostics
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU Detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "⚠️ Warning: No NVIDIA GPU detected. Running in CPU-only mode."
fi

# 2. Dependency Installations
echo "📦 Installing neural dependencies..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "websockets>=12.0" \
    "soundfile>=0.12.1" \
    "scipy>=1.13.0" \
    "numpy>=1.26.0,<2.0.0" \
    "pydantic>=2.8.0" \
    "httpx>=0.27.0" \
    "python-multipart>=0.0.9" \
    "faster-whisper>=1.0.0" \
    "onnxruntime>=1.17.0"

# Make log directory
mkdir -p logs

COMMON_PATH="services/common/src"

# 3. Launch Services with Dedicated Ports & Rotating Logging

# [0/5] Enhancement (ramo_clean) - Port 50054 (CPU / ONNX)
echo "🧹 [0/5] Starting Audio Enhancement Service on port 50054..."
PYTHONPATH="${COMMON_PATH}:services/enhancement/src" nohup python -m uvicorn ramo_clean.server:app \
    --host 0.0.0.0 --port 50054 > logs/enhancement_boot.log 2>&1 &
PID_CLEAN=$!

# [1/5] STT (ramo_listen) - Port 50051 (CUDA:0 ~1.5GB VRAM)
echo "👂 [1/5] Starting Speech-to-Text Service on port 50051..."
PYTHONPATH="${COMMON_PATH}:services/stt/src" nohup python -m uvicorn ramo_listen.server:app \
    --host 0.0.0.0 --port 50051 > logs/stt_boot.log 2>&1 &
PID_STT=$!

# [2/5] Diarization (ramo_speaker) - Port 50052 (CPU / CUDA:0 ~0.5GB VRAM)
echo "👥 [2/5] Starting Diarization & Harvester Service on port 50052..."
PYTHONPATH="${COMMON_PATH}:services/diarization/src" nohup python -m uvicorn ramo_speaker.server:app \
    --host 0.0.0.0 --port 50052 > logs/diarization_boot.log 2>&1 &
PID_DIAR=$!

# [3/5] Translation (ramo_translate) - Port 50053 (CUDA:0 ~2.0GB VRAM)
echo "🌐 [3/5] Starting Translation & Intelligence Service on port 50053..."
PYTHONPATH="${COMMON_PATH}:services/translation/src" nohup python -m uvicorn ramo_translate.server:app \
    --host 0.0.0.0 --port 50053 > logs/translation_boot.log 2>&1 &
PID_TRANS=$!

# [4/5] TTS (ramo_voice) - Port 50055 (CUDA:0 ~2.8GB VRAM / CPU Kokoro)
echo "🎙️ [4/5] Starting TTS & Voice Cloning Service on port 50055..."
PYTHONPATH="${COMMON_PATH}:services/tts/src" nohup python -m uvicorn ramo_voice.server:app \
    --host 0.0.0.0 --port 50055 > logs/tts_boot.log 2>&1 &
PID_TTS=$!

# [5/5] Gateway (ramo_gateway) - Port 50000 (Bridge-Tauri WebSocket Orchestrator)
echo "🚪 [5/5] Starting Sovereign Voice Gateway on port 50000..."
GATEWAY_PATH="${COMMON_PATH}:services/gateway/src:services/enhancement/src:services/stt/src:services/diarization/src:services/translation/src:services/tts/src"
PORT=50000 PYTHONPATH="${GATEWAY_PATH}" nohup python -m uvicorn ramo_gateway.server:app \
    --host 0.0.0.0 --port 50000 > logs/gateway_boot.log 2>&1 &
PID_GW=$!

# 4. Health Check Poller
echo ""
echo "⏳ Waiting for microservices to warm up and verify health..."
sleep 5

SERVICES=(
    "50054:Enhancement:ramo_clean"
    "50051:STT:ramo_listen"
    "50052:Diarization:ramo_speaker"
    "50053:Translation:ramo_translate"
    "50055:TTS:ramo_voice"
    "50000:Gateway:ramo_gateway"
)

ALL_HEALTHY=true

for item in "${SERVICES[@]}"; do
    IFS=":" read -r port name pkg <<< "$item"
    HEALTH_RES=$(curl -s "http://localhost:${port}/health" || true)
    if [[ "$HEALTH_RES" == *"healthy"* ]]; then
        echo "  ✅ Port ${port} [${name}]: Healthy"
    else
        echo "  ❌ Port ${port} [${name}]: Failed to report healthy! (Check logs/${pkg}.log)"
        ALL_HEALTHY=false
    fi
done

# 5. Status Dashboard
echo ""
echo "======================================================================"
echo "📊 ramO Sovereign Audio Intelligence Dashboard (Google Colab T4)"
echo "======================================================================"
printf "| %-5s | %-15s | %-8s | %-12s | %-16s |\n" "PORT" "SERVICE" "PID" "TARGET DEVICE" "VRAM ALLOCATION"
echo "|-------|-----------------|----------|--------------|------------------|"
printf "| %-5s | %-15s | %-8s | %-12s | %-16s |\n" "50054" "Enhancement" "$PID_CLEAN" "CPU (ONNX)" "0 MB"
printf "| %-5s | %-15s | %-8s | %-12s | %-16s |\n" "50051" "STT & Emotion" "$PID_STT" "CUDA:0 / CPU" "~1.5 GB"
printf "| %-5s | %-15s | %-8s | %-12s | %-16s |\n" "50052" "Diarization" "$PID_DIAR" "CPU / CUDA" "~0.5 GB"
printf "| %-5s | %-15s | %-8s | %-12s | %-16s |\n" "50053" "Translation" "$PID_TRANS" "CUDA:0 / CPU" "~2.0 GB"
printf "| %-5s | %-15s | %-8s | %-12s | %-16s |\n" "50055" "TTS Voice" "$PID_TTS" "CUDA:0 / CPU" "~2.8 GB"
printf "| %-5s | %-15s | %-8s | %-12s | %-16s |\n" "50000" "Voice Gateway" "$PID_GW" "CPU (AsyncIO)" "0 MB"
echo "======================================================================"
echo "🎯 Total Active VRAM: ~7.8 GB / 15.0 GB (Free Headroom: ~7.2 GB)"
echo ""
echo "🌐 Bridge-Tauri WebSocket URL: ws://localhost:50000/v1/stream"
echo "📡 Live Log Streamer:          ws://localhost:50000/v1/logs/stream"
echo "🎙️ OpenAI Speech (TTS):        http://localhost:50000/v1/audio/speech"
echo "👂 OpenAI STT Transcribe:      http://localhost:50000/v1/audio/transcriptions"
echo "🌐 Meeting Translation:        http://localhost:50000/v1/translate"
echo "======================================================================"

if [ "$ALL_HEALTHY" = true ]; then
    echo "🎉 All 6 Sovereign Services are LIVE and healthy!"
else
    echo "⚠️ One or more services failed health verification. Check logs for details."
fi
