#!/bin/bash
# =============================================================================
# deploy/run_colab.sh
# -----------------------------------------------------------------------------
# Execute this inside your WSL Ubuntu terminal:
#   bash deploy/run_colab.sh [colab_account_profile]
# Example:
#   bash deploy/run_colab.sh colab_acc_1
# =============================================================================

set -e

if [ -f .env ]; then
    source .env
    export $(grep -v '^#' .env | xargs)
else
    echo "[-] ERROR: .env file not found. Please create one with TAILSCALE_AUTHKEY, HF_TOKEN, and GITHUB_TOKEN."
    exit 1
fi

if [ -z "$TAILSCALE_AUTHKEY" ]; then
    echo "[-] ERROR: TAILSCALE_AUTHKEY is not set in .env!"
    exit 1
fi

# =============================================================================
# Pre-flight Network & DNS Resilience Check (Self-Healing)
# -----------------------------------------------------------------------------
ensure_dns_connectivity() {
    local test_cmd="import socket; socket.gethostbyname('oauth2.googleapis.com')"
    if ! python3 -c "$test_cmd" >/dev/null 2>&1; then
        echo "[!] Warning: DNS lookup failed for oauth2.googleapis.com (network change detected)."
        echo "[*] Attempting automatic DNS recovery using public fallbacks (8.8.8.8, 1.1.1.1)..."

        if [ -w /etc/resolv.conf ]; then
            printf "nameserver 8.8.8.8\nnameserver 1.1.1.1\n" >> /etc/resolv.conf 2>/dev/null || true
        elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
            sudo chattr -i /etc/resolv.conf 2>/dev/null || true
            printf "nameserver 8.8.8.8\nnameserver 1.1.1.1\n" | sudo tee -a /etc/resolv.conf >/dev/null 2>&1 || true
        fi

        if python3 -c "$test_cmd" >/dev/null 2>&1; then
            echo "[+] DNS restored successfully!"
        else
            echo "[-] ERROR: Unable to resolve internet domains. Please check your network connection."
            exit 1
        fi
    fi
}

ensure_dns_connectivity

echo "[*] Generating injected runner payload..."
cat << EOF > serve_ramo_injected.py
import os
os.environ["TAILSCALE_AUTHKEY"] = "$TAILSCALE_AUTHKEY"
os.environ["HF_TOKEN"] = "$HF_TOKEN"
os.environ["GITHUB_TOKEN"] = "${GITHUB_TOKEN:-}"
EOF

cat deploy/serve_ramo_colab.py >> serve_ramo_injected.py

# Account rotation logic
COLAB_ACC=${1:-"colab_acc_1"}
USER_HOME=$(eval echo "~")

TARGET_DIR=""
CANDIDATES=(
    "$COLAB_ACC"
    "${USER_HOME}/${COLAB_ACC}"
    "/home/iamomar/${COLAB_ACC}"
    "${USER_HOME}/.${COLAB_ACC}"
    "/home/iamomar/.${COLAB_ACC}"
)

for cand in "${CANDIDATES[@]}"; do
    if [ -d "$cand" ] && { [ -f "$cand/.config/colab-cli/token.json" ] || [ -f "$cand/token.json" ]; }; then
        TARGET_DIR="$cand"
        break
    elif [ -d "$cand" ] && [ -z "$TARGET_DIR" ]; then
        TARGET_DIR="$cand"
    fi
done

if [ -z "$TARGET_DIR" ] || [ ! -d "$TARGET_DIR" ]; then
    if [ -f "${USER_HOME}/.config/colab-cli/token.json" ] && [ -z "$1" ]; then
        TARGET_DIR="${USER_HOME}"
    else
        TARGET_DIR="${USER_HOME}/${COLAB_ACC}"
        mkdir -p "$TARGET_DIR"
        echo "[*] Auto-created Colab profile directory: $TARGET_DIR"
    fi
fi

COLAB_BIN=""
for b in "/home/iamomar/.local/bin/colab" "${USER_HOME}/.local/bin/colab" "$(which colab 2>/dev/null)"; do
    if [ -x "$b" ]; then
        COLAB_BIN="$b"
        break
    fi
done

if [ -z "$COLAB_BIN" ]; then
    COLAB_BIN="colab"
fi

echo "[*] Using Colab account profile: $TARGET_DIR"
echo "[*] Using colab-cli binary: $COLAB_BIN"

if [ -f "$TARGET_DIR/.config/colab-cli/token.json" ] || [ -f "$TARGET_DIR/token.json" ]; then
    HOME="$TARGET_DIR" "$COLAB_BIN" sessions > /dev/null 2>&1 || true
fi

echo "[*] Provisioning Colab T4 GPU and executing sovereign ramO Engine..."
HOME="$TARGET_DIR" "$COLAB_BIN" run --gpu T4 serve_ramo_injected.py

echo "[*] Cleaning up temporary payload..."
rm -f serve_ramo_injected.py
