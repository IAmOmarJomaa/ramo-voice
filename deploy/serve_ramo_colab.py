"""
deploy/serve_ramo_colab.py
==========================
Headless execution payload injected into Google Colab VM via colab-cli.
Handles Tailscale userspace mesh networking, dependency bootstrapping,
multi-service orchestration, and continuous foreground keep-alive telemetry.
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
    print("🚀 Sovereign ramO Engine: Colab T4 Deployment")
    print("=" * 70 + "\n", flush=True)

    # 1. Tailscale Mesh Setup (Userspace)
    if ts_authkey:
        print("[1/5] Starting Tailscale Mesh...", flush=True)
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
        print("[1/5] TAILSCALE_AUTHKEY not set — skipping mesh networking", flush=True)

    # 2. Fetch/Update Codebase if in Colab scratch directory
    if os.path.exists("/content") and not os.path.exists("services/gateway"):
        print("\n[2/5] Fetching ramo-engine codebase...", flush=True)
        if github_token:
            run_cmd(f"git clone --depth 1 https://oauth2:{github_token}@github.com/IAmOmarJomaa/ramo-engine.git /content/ramo-engine", check=False)
        else:
            run_cmd("git clone --depth 1 https://github.com/IAmOmarJomaa/ramo-engine.git /content/ramo-engine", check=False)
        if os.path.exists("/content/ramo-engine"):
            os.chdir("/content/ramo-engine")

    # 3. High-Speed Model Caching
    if hf_token:
        print("\n[3/5] Enabling HF Transfer acceleration...", flush=True)
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        run_cmd("pip install --quiet 'huggingface_hub[cli,hf_transfer]' hf_transfer", check=False)

    # 4. Boot All Sovereign Services via colab_run.sh
    print("\n[4/5] Booting Sovereign Services via deploy/colab_run.sh...", flush=True)
    run_cmd("chmod +x deploy/colab_run.sh", check=False)
    
    # Execute colab_run.sh in the foreground (it will tail logs and emit heartbeats)
    subprocess.run(["bash", "deploy/colab_run.sh"], env=os.environ.copy())


if __name__ == "__main__":
    main()
