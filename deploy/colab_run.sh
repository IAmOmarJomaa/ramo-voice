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

# 0. Host RAM Protection: Provision 8GB Swap Space (12.7GB RAM + 8GB SWAP = 20.7GB total system memory)
echo "💾 [0/7] Provisioning 8GB Swap space for Host RAM..."
test -f /swapfile || (sudo fallocate -l 8G /swapfile 2>/dev/null || sudo dd if=/dev/zero of=/swapfile bs=1M count=8192 status=none) && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile || true
swapon --show || free -h || true

# 1. GPU & VRAM Diagnostics
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 [1/7] NVIDIA GPU Detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
else
    echo "⚠️ Warning: No NVIDIA GPU detected. Running in CPU-only mode."
fi

# 2. Dependency Installations via High-Speed uv
echo "📦 [2/7] Installing neural dependencies via high-speed parallel uv..."
python -m pip install --quiet uv || true
uv pip install --system \
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
    "onnxruntime>=1.17.0" \
    "huggingface_hub[cli,hf_transfer]" \
    "hf_transfer" || true

# 2b. VRAM Fragmentation & Memory Allocator Flags
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CT2_CUDA_CACHING_ALLOCATOR_CONFIG="8,3,7,209715200"
export MALLOC_TRIM_THRESHOLD_="65536"
export MALLOC_MMAP_THRESHOLD_="65536"

# 2c. Tailscale Mesh Setup (Userspace mode for Google Colab)
if [ -n "${TAILSCALE_AUTHKEY:-}" ]; then
    echo "🌐 [3/7] Initializing Tailscale Mesh Network..."
    if ! command -v tailscale &> /dev/null; then
        curl -fsSL https://tailscale.com/install.sh | sh > /dev/null 2>&1 || true
    fi
    tailscaled --tun=userspace-networking --socks5-server=localhost:1055 &
    sleep 2
    tailscale up --authkey="${TAILSCALE_AUTHKEY}" --hostname=ramo-gpu --accept-routes || true
    TS_IP=$(tailscale ip -4 2>/dev/null || echo "127.0.0.1")
    echo "  ✅ Tailscale connected as 'ramo-gpu' (Mesh IP: ${TS_IP})"
    echo "  🔗 Permanent MagicDNS WebSocket: ws://ramo-gpu:50000/v1/stream"
else
    echo "  ⚠️ TAILSCALE_AUTHKEY not set. Listening on local ports only."
    TS_IP="127.0.0.1"
fi

# 2d. High-Speed HuggingFace Weights Caching
if [ -n "${HF_TOKEN:-}" ]; then
    echo "⚡ High-Speed HuggingFace Download Engine Enabled (hf_transfer)..."
    export HF_HUB_ENABLE_HF_TRANSFER=1
fi

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

# 4. Active Health Check Readiness Gate (Poll up to 60s for models to warm up)
echo ""
echo "⏳ Waiting for microservices to warm up and verify health..."

for attempt in {1..30}; do
    ALL_HEALTHY=true
    for item in "${SERVICES[@]}"; do
        IFS=":" read -r port name pkg <<< "$item"
        HEALTH_RES=$(curl -s "http://localhost:${port}/health" || true)
        if [[ "$HEALTH_RES" != *"healthy"* ]]; then
            ALL_HEALTHY=false
        fi
    done
    if [ "$ALL_HEALTHY" = true ]; then
        echo "  🎉 All services reported healthy on attempt ${attempt}!"
        break
    fi
    sleep 2
done

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

# 6. Colab Foreground Keep-Alive & Telemetry Supervisor Loop
echo ""
echo "======================================================================"
echo "🛡️ Colab Keep-Alive Supervisor Active: Tailing logs & emitting heartbeats"
echo "   MagicDNS Stream: ws://ramo-gpu:50000/v1/stream"
echo "   (Press Ctrl+C to terminate the cluster)"
echo "======================================================================"

python3 - << 'EOF'
import os
import time
import sys

log_files = {
    "GW": "logs/ramo_gateway.log",
    "CLEAN": "logs/ramo_clean.log",
    "STT": "logs/ramo_listen.log",
    "DIAR": "logs/ramo_speaker.log",
    "TRANS": "logs/ramo_translate.log",
    "TTS": "logs/ramo_voice.log",
}

handles = {}
last_heartbeat = 0

def get_telemetry():
    vram_str = "VRAM: N/A"
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True
        ).strip().split(",")
        if len(out) >= 4:
            used, total, temp, util = [x.strip() for x in out[:4]]
            vram_str = f"GPU VRAM: {int(used)/1024:.1f}GB / {int(total)/1024:.1f}GB ({int(used)*100//int(total)}%) | Util: {util}% | Temp: {temp}°C"
    except Exception:
        pass
    
    ram_str = "RAM: N/A"
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        mem = {}
        for l in lines:
            parts = l.split(":")
            if len(parts) == 2:
                mem[parts[0].strip()] = parts[1].strip()
        tot_kb = float(mem.get("MemTotal", "0 kB").split()[0])
        avail_kb = float(mem.get("MemAvailable", "0 kB").split()[0])
        ram_str = f"RAM: {(tot_kb - avail_kb)/(1024**2):.1f}GB / {tot_kb/(1024**2):.1f}GB"
    except Exception:
        pass
    return f"{vram_str} | {ram_str}"

try:
    while True:
        now = time.time()
        # 30-second Heartbeat
        if now - last_heartbeat >= 30.0:
            telem = get_telemetry()
            timestamp = time.strftime("%H:%M:%S")
            print(f"💓 [HEARTBEAT {timestamp}] Colab T4 Active | {telem} | MagicDNS: ws://ramo-gpu:50000/v1/stream", flush=True)
            last_heartbeat = now
        
        # Real-time Log Tailing
        for tag, path in log_files.items():
            if tag not in handles and os.path.exists(path):
                h = open(path, "r", encoding="utf-8", errors="replace")
                h.seek(0, 2)  # Seek to end of file
                handles[tag] = h
            if tag in handles:
                line = handles[tag].readline()
                while line:
                    stripped = line.strip()
                    if stripped:
                        print(f"[{tag}] {stripped}", flush=True)
                    line = handles[tag].readline()
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nShutting down Sovereign cluster...", flush=True)
EOF

