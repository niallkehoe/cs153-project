# deploy/

Infrastructure for serving GPT-1900 on a DigitalOcean GPU droplet.

## Overview

GPT-1900 (3.3B parameters) requires a GPU for inference. In bf16, the model
weights occupy ~6.6GB of VRAM. A DigitalOcean GPU Droplet with an H100 80GB
or NVIDIA L40S 48GB is more than sufficient.

**Estimated cost:** ~$1.40–$2.00/hr (L40S). Spin up only during eval runs.

## Setup

### 1. Create the droplet

```bash
# Via doctl CLI
doctl compute droplet create gpt1900-server \
  --size gpu-h100x1-80gb \
  --image ubuntu-22-04-x64 \
  --region nyc3 \
  --ssh-keys <your-key-id>
```

Or use the DigitalOcean web console. Select a GPU Droplet under "All Droplets".

### 2. Install dependencies on the droplet

```bash
ssh root@<droplet-ip>

# Clone the GPT-1900 repo
git clone https://github.com/michaelhla/gpt1900.git
cd gpt1900
uv sync --extra gpu
source .venv/bin/activate

# Download the model (chat.sh handles this)
bash runs/chat.sh --download-only

# Install the server deps
pip install fastapi uvicorn httpx
```

### 3. Clone this project and start the server

```bash
git clone https://github.com/<your-org>/cs153-project.git
cd cs153-project
python deploy/server.py --model-dir /root/gpt1900 --port 8000
```

### 4. Run eval from your local machine

```bash
python eval/runner.py \
  --scientist-url http://<droplet-ip>:8000 \
  --eval eval/eval_set.json \
  --out results/runs/
```

### 5. Destroy the droplet when done

```bash
doctl compute droplet delete gpt1900-server
```

## Server API

`server.py` exposes a single endpoint:

```
POST /generate
Content-Type: application/json

{
  "prompt": "<full prompt string>",
  "temperature": 0.7,
  "top_k": 50,
  "max_new_tokens": 512
}

→ { "text": "<generated text>" }
```

## Notes

- The nanochat generate loop runs on a single H100/L40S. Latency per turn is
  typically 2–5 seconds for 512 tokens.
- The server is intentionally minimal — no batching, no streaming. Batching
  is not needed for sequential eval turns.
- Set `CUDA_VISIBLE_DEVICES=0` if the droplet has multiple GPUs.
