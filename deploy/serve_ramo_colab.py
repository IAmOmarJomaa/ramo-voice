"""
deploy/serve_ramo_colab.py
==========================
Pure-Python Sovereign Execution Payload for Google Colab T4 GPU.
Bootstraps Host RAM Swap, Tailscale Mesh, High-Speed uv Dependencies,
Phased Service Warmup, Port Health Checks, and Live Telemetry Streaming.
"""

import os
import sys
import time
import json
import socket
import subprocess
from pathlib import Path


def run_cmd(cmd: str, check: bool = True, timeout: int = 600) -> subprocess.CompletedProcess:
    print(f"[*] Running: {cmd}", flush=True)
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(cmd, shell=True, check=check, timeout=timeout, env=env)


def check_socket(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def get_system_telemetry() -> str:
    vram_str = "VRAM: N/A"
    try:
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
        ram_str = f"Host RAM: {(tot_kb - avail_kb)/(1024**2):.1f}GB / {tot_kb/(1024**2):.1f}GB"
    except Exception:
        pass
    return f"{vram_str} | {ram_str}"


def main():
    ts_authkey = os.environ.get("TAILSCALE_AUTHKEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    github_token = os.environ.get("GITHUB_TOKEN", "")

    print("\n" + "=" * 70, flush=True)
    print("🚀 Sovereign ramO Engine: Full Colab T4 Deployment", flush=True)
    print("=" * 70 + "\n", flush=True)

    # 1. Host RAM Protection: Provision 8GB Swap Space (12.7GB RAM + 8GB SWAP = 20.7GB total)
    print("[1/6] Provisioning 8GB Swap Space for Host RAM...", flush=True)
    run_cmd(
        "test -f /swapfile || (sudo fallocate -l 8G /swapfile 2>/dev/null || sudo dd if=/dev/zero of=/swapfile bs=1M count=8192 status=none) && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile || true",
        check=False
    )
    run_cmd("free -h", check=False)

    # 2. GPU & VRAM Diagnostics (15.0 GB NVIDIA T4)
    print("\n[2/6] GPU & VRAM Diagnostics...", flush=True)
    run_cmd("nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader", check=False)

    # 3. Tailscale Mesh Setup (Userspace)
    ts_ip = "127.0.0.1"
    if ts_authkey:
        print("\n[3/6] Starting Tailscale Mesh...", flush=True)
        run_cmd("command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh", check=False)
        subprocess.Popen(
            ["tailscaled", "--tun=userspace-networking", "--socks5-server=localhost:1055"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        run_cmd(f"tailscale up --authkey={ts_authkey} --hostname=ramo-gpu --accept-routes", check=False)
        try:
            ts_ip = subprocess.check_output("tailscale ip -4", shell=True).decode().strip()
            print(f"  ✅ Tailscale connected! Node IP: {ts_ip}", flush=True)
            print(f"  🔗 Permanent MagicDNS URL: ws://ramo-gpu:50000/v1/stream", flush=True)
        except Exception:
            print("  ⚠️ Could not read Tailscale IP, continuing...", flush=True)
    else:
        print("\n[3/6] TAILSCALE_AUTHKEY not set — skipping mesh networking", flush=True)

    # 4. Fetch/Update Codebase unconditionally in current directory
    print("\n[4/6] Fetching latest ramo-voice codebase...", flush=True)
    branch = os.environ.get("RAMO_BRANCH", "master")
    run_cmd("rm -rf ramo-voice", check=False)
    if github_token:
        tarball_cmd = f"mkdir -p ramo-voice && curl -sL -H 'Authorization: token {github_token}' -H 'User-Agent: Mozilla/5.0' https://api.github.com/repos/IAmOmarJomaa/ramo-voice/tarball/{branch} | tar -xz -C ramo-voice --strip-components=1"
        res = subprocess.run(tarball_cmd, shell=True)
        if res.returncode != 0 or not os.path.exists("ramo-voice/services"):
            run_cmd(f"git clone --depth 1 --branch {branch} --single-branch https://oauth2:{github_token}@github.com/IAmOmarJomaa/ramo-voice.git ramo-voice", check=False)
    else:
        run_cmd(f"git clone --depth 1 --branch {branch} --single-branch https://github.com/IAmOmarJomaa/ramo-voice.git ramo-voice", check=False)

    if os.path.exists("ramo-voice"):
        os.chdir("ramo-voice")
        print(f"  ✅ Entered working directory: {os.getcwd()}", flush=True)

    # 5. High-Speed Parallel uv Dependencies Installation
    print("\n[5/6] Installing Neural Dependencies via high-speed parallel uv...", flush=True)
    run_cmd("pip install uv", check=False)
    
    # Pre-pinning and installing core packages in parallel
    run_cmd(
        'uv pip install --system "fastapi>=0.115.0" "uvicorn[standard]>=0.30.0" "websockets>=12.0" "soundfile>=0.12.1" "scipy>=1.13.0" "numpy>=1.26.0,<2.0.0" "pydantic>=2.8.0" "httpx>=0.27.0" "python-multipart>=0.0.9" "faster-whisper>=1.0.0" "onnxruntime>=1.17.0" "huggingface_hub[cli,hf_transfer]" hf_transfer transformers accelerate'
    )

    # Provision CampPlus Diarization Model
    print("  🧠 Provisioning 3D-CAM++ (CampPlus) Diarization ONNX weights...", flush=True)
    run_cmd(
        "mkdir -p models && (test -f models/campplus.onnx || wget -q -c -O models/campplus.onnx https://huggingface.co/Luigi/campplus-zh-en-onnx/resolve/main/campplus_zh_en_fp32.onnx)",
        check=False
    )

    # 6. Memory Allocator & Environment Flags
    worker_env = os.environ.copy()
    worker_env["PYTHONUNBUFFERED"] = "1"
    worker_env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    worker_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    worker_env["CT2_CUDA_CACHING_ALLOCATOR_CONFIG"] = "8,3,7,209715200"
    worker_env["MALLOC_TRIM_THRESHOLD_"] = "65536"
    worker_env["MALLOC_MMAP_THRESHOLD_"] = "65536"
    worker_env["RAMO_LOAD_NEURAL_LLM"] = "1"
    worker_env["RAMO_LLM_MODEL"] = os.getenv("RAMO_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

    # Make log directory
    os.makedirs("logs", exist_ok=True)
    python_bin = sys.executable

    common_path = os.path.abspath("services/common/src")

    services_config = [
        ("Enhancement", 50054, "ramo_clean", f"{common_path}:services/enhancement/src", "ramo_clean.server:app"),
        ("STT", 50051, "ramo_listen", f"{common_path}:services/stt/src", "ramo_listen.server:app"),
        ("Diarization", 50052, "ramo_speaker", f"{common_path}:services/diarization/src", "ramo_speaker.server:app"),
        ("Translation", 50053, "ramo_translate", f"{common_path}:services/translation/src", "ramo_translate.server:app"),
        ("TTS", 50055, "ramo_voice", f"{common_path}:services/tts/src", "ramo_voice.server:app"),
        ("Gateway", 50000, "ramo_gateway", f"{common_path}:services/gateway/src:services/enhancement/src:services/stt/src:services/diarization/src:services/translation/src:services/tts/src", "ramo_gateway.server:app"),
    ]

    print("\n[6/6] Launching Sovereign Microservices Swarm...", flush=True)
    procs = {}

    for name, port, pkg, pypath, app_str in services_config:
        print(f"  🚀 Booting {name} Service on port {port}...", flush=True)
        env = worker_env.copy()
        env["PYTHONPATH"] = pypath
        env["PORT"] = str(port)

        log_path = f"logs/{pkg}.log"
        log_file = open(log_path, "a", encoding="utf-8")

        proc = subprocess.Popen(
            [python_bin, "-m", "uvicorn", app_str, "--host", "0.0.0.0", "--port", str(port)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        procs[name] = (proc, port, log_path)

    # Active Readiness Gate: Wait for ports to become open
    print("\n⏳ Active Readiness Gate: Waiting for all 6 microservices to report healthy...", flush=True)
    ready_ports = set()
    all_ports = {port for _, port, _, _, _ in services_config}

    for attempt in range(60):
        time.sleep(1)
        for name, port, _, _, _ in services_config:
            if port not in ready_ports and check_socket(port):
                ready_ports.add(port)
                print(f"  ✅ {name} on port {port} is SERVING & HEALTHY! ({len(ready_ports)}/6 ready)", flush=True)
        if len(ready_ports) == len(all_ports):
            break
        if attempt > 0 and attempt % 10 == 0:
            print(f"     ... warming up neural weights ({len(ready_ports)}/6 ready, {attempt}s elapsed)...", flush=True)

    if len(ready_ports) == len(all_ports):
        print("\n🎉 ALL 6 SOVEREIGN SERVICES ARE WARMED UP & SERVING ON PORT 50000!\n", flush=True)
    else:
        print(f"\n⚠️ Swarm booted with {len(ready_ports)}/6 services ready. Checking crashed logs...\n", flush=True)
        for name, (p, port, log_p) in procs.items():
            if port not in ready_ports:
                print(f"[-] Log snippet for {name} ({log_p}):", flush=True)
                if os.path.exists(log_p):
                    with open(log_p, "r", encoding="utf-8", errors="replace") as lf:
                        print(lf.read()[-1000:], flush=True)

    # Status Dashboard
    print("=" * 70, flush=True)
    print("📊 ramO Sovereign Audio Intelligence Dashboard (Google Colab T4)", flush=True)
    print("=" * 70, flush=True)
    print(f"🌐 MagicDNS WebSocket: ws://ramo-gpu:50000/v1/stream", flush=True)
    print(f"📡 Direct IP WebSocket: ws://{ts_ip}:50000/v1/stream", flush=True)
    print(f"📋 Log Buffet (HTTP):   http://{ts_ip}:9090/ (Click to inspect all logs)", flush=True)
    print("=" * 70 + "\n", flush=True)

    # Spawn Log Buffet HTTP Server on port 9090
    subprocess.Popen(
        [python_bin, "-m", "http.server", "9090", "--directory", "logs"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )

    # Live Log Streamer & Telemetry Heartbeat Loop
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

    print("📋 Live Telemetry & Traffic Stream (Tailing logs in real time):", flush=True)
    print("=======================================================\n", flush=True)

    try:
        while True:
            now = time.time()
            if now - last_heartbeat >= 30.0:
                telem = get_system_telemetry()
                timestamp = time.strftime("%H:%M:%S")
                print(f"💓 [HEARTBEAT {timestamp}] Colab T4 Active | {telem} | ws://ramo-gpu:50000/v1/stream", flush=True)
                last_heartbeat = now

            for tag, path in log_files.items():
                if tag not in handles and os.path.exists(path):
                    h = open(path, "r", encoding="utf-8", errors="replace")
                    h.seek(0, 2)
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
        print("\nShutting down Sovereign node...", flush=True)


if __name__ == "__main__":
    main()
