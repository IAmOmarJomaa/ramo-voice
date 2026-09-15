"""
deploy/serve_ramo_colab.py
==========================
Headless execution payload injected into Google Colab VM via colab-cli.
Handles Host RAM swap provisioning, Tailscale mesh networking, uv parallel
dependency bootstrapping, and full multi-service keep-alive supervision.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path


def run_cmd(cmd: str, check: bool = True, timeout: int = 600) -> subprocess.CompletedProcess:
    print(f"[*] Running: {cmd}", flush=True)
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(cmd, shell=True, check=check, timeout=timeout, env=env)


def main():
    ts_authkey = os.environ.get("TAILSCALE_AUTHKEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    github_token = os.environ.get("GITHUB_TOKEN", "")

    print("\n" + "=" * 70)
    print("🚀 Sovereign ramO Engine: Colab T4 Cloud Deployment")
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
            print(f"  🔗 MagicDNS URL: ws://ramo-gpu:50000/v1/stream", flush=True)
        except Exception:
            print("  ⚠️ Could not read Tailscale IP, continuing...", flush=True)
    else:
        print("\n[3/6] TAILSCALE_AUTHKEY not set — skipping mesh networking", flush=True)

    # 4. Fetch/Update Codebase if in Colab scratch directory
    if os.path.exists("/content") and not os.path.exists("services/gateway"):
        branch = os.environ.get("RAMO_BRANCH", "master")
        print(f"\n[4/6] Fetching ramo-voice codebase (branch: {branch})...", flush=True)
        if github_token:
            tarball_cmd = f"mkdir -p ramo-voice && curl -sL -H 'Authorization: token {github_token}' -H 'User-Agent: Mozilla/5.0' https://api.github.com/repos/IAmOmarJomaa/ramo-voice/tarball/{branch} | tar -xz -C ramo-voice --strip-components=1"
            res = subprocess.run(tarball_cmd, shell=True)
            if res.returncode != 0 or not os.path.exists("ramo-voice/services"):
                run_cmd(f"git clone --depth 1 --branch {branch} --single-branch https://oauth2:{github_token}@github.com/IAmOmarJomaa/ramo-voice.git ramo-voice", check=False)
        else:
            run_cmd(f"git clone --depth 1 --branch {branch} --single-branch https://github.com/IAmOmarJomaa/ramo-voice.git ramo-voice", check=False)
        if os.path.exists("ramo-voice"):
            os.chdir("ramo-voice")

    # 5. High-Speed Model Caching
    if hf_token:
        print("\n[5/6] Enabling HF Transfer acceleration...", flush=True)
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # 6. Boot All Sovereign Services via colab_run.sh (with unbuffered live stdout)
    print("\n[6/6] Booting Sovereign Services via deploy/colab_run.sh...", flush=True)
    run_cmd("chmod +x deploy/colab_run.sh", check=False)
    
    # Run colab_run.sh unbuffered in the foreground
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(["bash", "deploy/colab_run.sh"], env=child_env)


if __name__ == "__main__":
    main()
