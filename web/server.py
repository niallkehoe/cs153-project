"""
web/server.py
-------------
FastAPI server for the Machina Mirabilis frontend.

Serves the single-page frontend and provides API endpoints for scenario data
and demo playback. Designed to be extended with real pipeline execution in
Phase 2 (using OPEN_ROUTER_API_KEY).

Usage
~~~~~
  cd /path/to/cs153-project
  mamba activate sylvian
  pip install fastapi uvicorn
  uvicorn web.server:app --reload --port 7860

Endpoints
~~~~~~~~~
  GET  /                          → index.html
  GET  /api/scenarios             → list of eval scenarios (metadata only)
  GET  /api/demo/{scenario_id}    → pre-recorded demo run for a scenario
  GET  /api/health                → server health check
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent
EVAL_SET_PATH = BASE_DIR.parent / "eval" / "eval_set.json"
DEMO_DATA_PATH = BASE_DIR / "demo_data.json"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Machina Mirabilis", version="1.0.0")

# ------------------------------------------------------------------
# Load data at startup
# ------------------------------------------------------------------

with open(EVAL_SET_PATH) as f:
    _EVAL_SET: list[dict] = json.load(f)

with open(DEMO_DATA_PATH) as f:
    _DEMO_DATA: list[dict] = json.load(f)

_DEMO_INDEX: dict[str, dict] = {s["scenario_id"]: s for s in _DEMO_DATA}

# ------------------------------------------------------------------
# Static files
# ------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "scenarios": len(_EVAL_SET)})


@app.get("/api/scenarios")
async def scenarios() -> JSONResponse:
    """Return eval scenario metadata (no ground truth leaked)."""
    result = [
        {
            "id": e["id"],
            "tier": e.get("tier"),
            "domain": e.get("domain"),
            "difficulty": e.get("difficulty"),
            "max_turns": e.get("max_turns"),
            "question": e["question"],
        }
        for e in _EVAL_SET
    ]
    return JSONResponse(result)


@app.get("/api/demo/{scenario_id}")
async def demo_run(scenario_id: str) -> JSONResponse:
    """Return the pre-recorded demo conversation for a scenario."""
    if scenario_id not in _DEMO_INDEX:
        available = list(_DEMO_INDEX.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Demo not found for '{scenario_id}'. Available: {available}",
        )
    return JSONResponse(_DEMO_INDEX[scenario_id])
