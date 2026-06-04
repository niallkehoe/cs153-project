# deploy/

Infrastructure for serving GPT-1900 on a GCP GPU instance.

## Overview

GPT-1900 (3.3B parameters) requires a GPU for inference. In bf16, the model
weights occupy ~6.6GB of VRAM. The current setup uses an NVIDIA L4 (24GB) on
Google Cloud Platform, accessed locally via an SSH tunnel.

**VM:** `gpu-l4-1`, zone `us-central1-c`, project `cs229-497921`

## Setup

### 1. Download the model on the VM

```bash
gcloud compute ssh gpu-l4-1 --zone=us-central1-c --project=cs229-497921

# On the VM:
git clone https://github.com/michaelhla/gpt1900.git ~/gpt1900
cd ~/gpt1900
uv sync --extra gpu
source .venv/bin/activate

# Download model weights to a separate directory
NANOCHAT_BASE_DIR=$HOME/gpt1900_models bash ~/gpt1900/runs/chat.sh --download-only
```

### 2. Copy the server to the VM

```bash
gcloud compute scp deploy/server.py gpu-l4-1:~/cs153_server.py \
  --zone=us-central1-c --project=cs229-497921
```

### 3. Start the server on the VM

```bash
gcloud compute ssh gpu-l4-1 --zone=us-central1-c --project=cs229-497921

# On the VM:
python ~/cs153_server.py \
  --model-dir ~/gpt1900 \
  --model-files-dir ~/gpt1900_models/gpt1900-instruct-v3-sft \
  --host 127.0.0.1 \
  --port 8000
```

### 4. Open the SSH tunnel locally (keep this running)

```bash
gcloud compute ssh gpu-l4-1 --zone=us-central1-c --project=cs229-497921 \
  -- -NL 8000:localhost:8000
```

The GPT-1900 server is now reachable at `http://localhost:8000` on your local machine.

### 5. Verify the connection

```bash
curl http://localhost:8000/health
# → {"status":"ok","model_loaded":"True","device":"cuda"}

curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The laws of motion state that", "max_new_tokens": 50}'
```

## Server API

`server.py` exposes two endpoints:

```
GET /health
→ {"status": "ok", "model_loaded": "True", "device": "cuda"}

POST /generate
Content-Type: application/json
{
  "prompt":         "<full prompt string>",
  "temperature":    0.6,
  "top_k":          20,
  "max_new_tokens": 256,
  "stop_sequences": ["\nEND", " END"]   # optional; trimmed server-side
}
→ {"text": "<generated text>"}
```

## Notes

- Latency per turn is typically 5–15 seconds for 256 tokens on the L4.
- The server binds to `127.0.0.1` only; the SSH tunnel exposes it locally.
- Repetition detection and stop-sequence trimming run server-side before the
  response is returned, so the client always receives clean text.
- Set `CUDA_VISIBLE_DEVICES=0` if the instance has multiple GPUs.
