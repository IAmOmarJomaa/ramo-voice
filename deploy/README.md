# 🚀 Google Colab T4 Deployment Guide

This guide explains how to run the entire sovereign ramO Engine on a standard **Google Colab T4 GPU (15GB VRAM)** with zero out-of-memory crashes and zero thermal throttling.

---

## 💾 T4 VRAM Budget Allocation

| Service | Subsystem | Target Device | VRAM Allocation |
| :--- | :--- | :--- | :--- |
| **`services/gateway`** | Turn-taking state machine & VAD | CPU | 0 MB |
| **`services/tts`** | Chunker & Equal-Power Crossfader | CPU | 0 MB |
| **`services/tts`** | Supertonic ONNX Fast-Path (<90ms TTFA) | CPU | 0 MB |
| **`services/tts`** | Flow Matching Zero-Shot Cloner | GPU (CUDA:0) | ~3.8 GB |
| **`services/stt`** | SenseVoice + LocalAgreement Streaming | GPU (CUDA:0) | ~2.2 GB |
| **`services/diarization`** | Speaker Diarization + Voiceprint Harvester | GPU (CUDA:0) | ~1.4 GB |
| **System Headroom** | CUDA Runtime & Memory Cache | GPU | ~1.0 GB |
| **FREE HEADROOM** | Available for 8B Quantized LLM or safety buffer | GPU | **~6.6 GB** |

---

## ⚡ Quick Start on Google Colab

Run this single cell in your Google Colab Notebook:

```python
# 1. Clone your repo
!git clone https://github.com/iamomarjomaa/ramo-engine.git /content/ramo-engine
%cd /content/ramo-engine

# 2. Run the deployment script
!bash deploy/colab_run.sh
```

---

## 🌐 Exposing Endpoints Publicly

To connect your local client or web UI to the Colab server, use Cloudflare Tunnels or Ngrok:

```python
!pip install pyngrok
from pyngrok import ngrok

# Expose the Gateway port (50050)
gateway_tunnel = ngrok.connect(50050)
print("Public Gateway URL:", gateway_tunnel.public_url)

# Expose the TTS port (50055)
tts_tunnel = ngrok.connect(50055)
print("Public TTS URL:", tts_tunnel.public_url)
```
