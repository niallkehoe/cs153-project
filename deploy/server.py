"""
server.py
---------
Minimal FastAPI wrapper around the GPT-1900 nanochat generation loop.

Exposes a single POST /generate endpoint that the ScientistAgent client
calls for each turn in the experiment loop.

This server is designed to run on a DigitalOcean GPU droplet alongside the
cloned gpt1900 repository. It imports directly from nanochat (the gpt1900
framework) rather than bundling it.

See deploy/README.md for full setup instructions.

Usage
~~~~~
  python deploy/server.py \\
    --model-dir /path/to/gpt1900 \\
    --checkpoint runs/chat.sh (default model) \\
    --port 8000

Endpoint
~~~~~~~~
  POST /generate
  {
    "prompt": "<full prompt string>",
    "temperature": 0.7,
    "top_k": 50,
    "max_new_tokens": 512
  }
  → { "text": "<generated text>" }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""

    prompt: str = Field(..., description="Full prompt string including conversation history.")
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_k: int = Field(default=20, ge=1, le=500)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    stop_sequences: list[str] = Field(
        default_factory=list,
        description="Stop generation when any of these strings appear in the output.",
    )


class GenerateResponse(BaseModel):
    """Response body from the /generate endpoint."""

    text: str = Field(..., description="Generated text (continuation only, not including prompt).")


def _apply_stop_sequences(text: str, stop_sequences: list[str]) -> str:
    """Trim the generated text at the first occurrence of any stop sequence."""
    for seq in stop_sequences:
        idx = text.find(seq)
        if idx >= 0:
            text = text[: idx + len(seq)]
    return text


def _truncate_repetition(text: str, ngram: int = 4, max_reps: int = 4) -> str:
    """Return text truncated just before a repeating N-gram loop begins."""
    words = text.split()
    if len(words) < ngram * max_reps:
        return text
    seen: dict[tuple, list[int]] = {}
    for i in range(len(words) - ngram + 1):
        gram = tuple(words[i : i + ngram])
        positions = seen.setdefault(gram, [])
        positions.append(i)
        if len(positions) >= max_reps:
            cut = positions[max_reps - 1]
            return " ".join(words[:cut]).strip()
    return text


app = FastAPI(
    title="GPT-1900 Inference Server",
    description="Serves the GPT-1900 model for the CS153 re-discovery eval pipeline.",
    version="0.1.0",
)

# Global model state — loaded once at startup
_model = None
_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(
    model_dir: str,
    checkpoint: str | None = None,
    model_files_dir: str | None = None,
) -> None:
    """
    Load GPT-1900 model and tokenizer from the nanochat model directory.

    Expects the gpt1900 repo to have been cloned at ``model_dir`` and the
    model files to have been downloaded (via ``bash runs/chat.sh``).

    File discovery order
    ~~~~~~~~~~~~~~~~~~~~
    - Tokenizer: ``{model_files_dir}/tokenizer/``
    - Meta JSON: explicit ``checkpoint`` path stripped to ``.json``, or the
      first ``meta_*.json`` found in ``model_files_dir``
    - Checkpoint: explicit ``checkpoint`` path if given, else the first
      ``model_*.pt`` found in ``model_files_dir``

    Parameters
    ----------
    model_dir:
        Path to the cloned gpt1900 repository root (added to sys.path so
        ``nanochat`` can be imported).
    checkpoint:
        Optional path to a specific ``.pt`` checkpoint file. If None, the
        most recently modified ``model_*.pt`` in ``model_files_dir`` is used.
    model_files_dir:
        Directory containing ``tokenizer/``, ``meta_*.json``, and
        ``model_*.pt``. Defaults to ``model_dir`` when not set, which is
        correct when the model files live inside the cloned repo.
    """
    global _model, _tokenizer

    root = Path(model_files_dir or model_dir)

    # Import nanochat from the gpt1900 repo (inserted into sys.path by main())
    from nanochat.gpt import GPT, GPTConfig  # type: ignore[import]
    from nanochat.tokenizer import RustBPETokenizer  # type: ignore[import]

    # --- Tokenizer ---
    tokenizer_dir = root / "tokenizer"
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")
    _tokenizer = RustBPETokenizer.from_directory(str(tokenizer_dir))

    # --- Meta JSON ---
    if checkpoint:
        meta_path = Path(checkpoint).with_suffix(".json")
        # nanochat names meta files like meta_XXXXXX.json alongside model_XXXXXX.pt
        if not meta_path.exists():
            stem = Path(checkpoint).stem.replace("model_", "meta_")
            meta_path = root / f"{stem}.json"
    else:
        candidates = sorted(root.glob("meta_*.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No meta_*.json found in {root}")
        meta_path = candidates[-1]

    with open(meta_path) as f:
        meta = json.load(f)

    # --- Checkpoint ---
    if checkpoint:
        ckpt_path = Path(checkpoint)
    else:
        candidates = sorted(root.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No model_*.pt found in {root}")
        ckpt_path = candidates[-1]

    print(f"Loading meta  : {meta_path}")
    print(f"Loading ckpt  : {ckpt_path}")
    print(f"Device        : {_device}")

    config = GPTConfig(**meta["model_config"])
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=_device)
    model.init_weights()

    state_dict = torch.load(str(ckpt_path), map_location=_device)
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.eval()

    _model = model
    print("Model loaded.")


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Run a single forward pass and return the generated continuation.

    The prompt should include the full conversation history formatted by
    ScientistAgent._build_prompt(). The response contains only the newly
    generated tokens, not the prompt.

    Raises
    ------
    HTTPException (503)
        If the model has not been loaded yet.
    HTTPException (500)
        If generation fails for any reason.
    """
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Start server with --model-dir.")

    try:
        bos = _tokenizer.get_bos_token_id()
        token_ids: list[int] = _tokenizer.encode(request.prompt, prepend=bos)

        generated: list[int] = []
        autocast_device = "cuda" if _device == "cuda" else "cpu"
        with torch.amp.autocast(device_type=autocast_device, dtype=torch.bfloat16):
            for token in _model.generate(
                token_ids,
                max_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
            ):
                generated.append(token)

        text = _tokenizer.decode(generated)
        text = _apply_stop_sequences(text, request.stop_sequences)
        text = _truncate_repetition(text)
        return GenerateResponse(text=text)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint. Returns model load status."""
    return {
        "status": "ok",
        "model_loaded": str(_model is not None),
        "device": _device,
    }


def main() -> None:
    """Parse CLI args, load model, start server."""
    parser = argparse.ArgumentParser(description="GPT-1900 inference server.")
    parser.add_argument("--model-dir", required=True, help="Path to gpt1900 repo root (for nanochat imports)")
    parser.add_argument(
        "--model-files-dir",
        default=None,
        help="Directory containing tokenizer/, meta_*.json, model_*.pt. "
             "Defaults to --model-dir when not set.",
    )
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint .pt file")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    sys.path.insert(0, args.model_dir)
    load_model(args.model_dir, args.checkpoint, args.model_files_dir)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
