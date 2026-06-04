"""
web/server.py
-------------
FastAPI server for the Machina Mirabilis frontend.

Serves the single-page frontend and provides API endpoints for scenario
metadata and live experiment runs streamed via Server-Sent Events (SSE).

Usage
~~~~~
  cd /path/to/cs153-project
  mamba activate sylvian
  uvicorn web.server:app --reload --port 7860

Environment variables (loaded from .env at startup)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  OPEN_ROUTER_API_KEY   — OpenRouter API key for simulator and judge
  SCIENTIST_API_URL     — Base URL for the GPT-1900 server (default: http://localhost:8000)
  SIMULATOR_MODEL       — OpenRouter model for simulator (default: anthropic/claude-3.5-haiku)
  JUDGE_MODEL           — OpenRouter model for judge (default: anthropic/claude-3.5-haiku)

Endpoints
~~~~~~~~~
  GET  /                          → index.html
  GET  /api/scenarios             → list of eval scenarios (metadata only)
  GET  /api/run/{scenario_id}     → SSE stream of live experiment events
  GET  /api/health                → server health check
"""

from __future__ import annotations

import json
import os
import queue
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Load .env from project root before importing agents (they read os.environ)
load_dotenv(Path(__file__).parent.parent / ".env")

# Add project root to sys.path so agent imports resolve regardless of CWD
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.judge import JudgeAgent
from agents.scientist import ScientistAgent
from agents.simulator import SimulatorAgent
from tools.router import LoopAction, ToolRouter
from tools.schema import ConcludeArgs

BASE_DIR = Path(__file__).parent
EVAL_SET_PATH = BASE_DIR.parent / "eval" / "eval_set.json"
STATIC_DIR = BASE_DIR / "static"

SCIENTIST_API_URL = os.getenv("SCIENTIST_API_URL", "http://localhost:8000")
SIMULATOR_MODEL = os.getenv("SIMULATOR_MODEL", "anthropic/claude-3.5-haiku")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "anthropic/claude-3.5-haiku")

app = FastAPI(title="Machina Mirabilis", version="2.0.0")

with open(EVAL_SET_PATH) as f:
    _EVAL_SET: list[dict] = json.load(f)

_EVAL_INDEX: dict[str, dict] = {e["id"]: e for e in _EVAL_SET}

_TOOL_BLOCK_RE = re.compile(
    r"TOOL:\s*(?P<name>\w+)\s*\n(?P<body>.*?)END",
    re.DOTALL | re.IGNORECASE,
)

_executor = ThreadPoolExecutor(max_workers=4)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ------------------------------------------------------------------
# Pipeline helpers
# ------------------------------------------------------------------

def _parse_scientist_output(raw: str) -> dict:
    """Extract prose and tool call from a raw scientist generation."""
    match = _TOOL_BLOCK_RE.search(raw)
    if match:
        prose = raw[: match.start()].strip()
        tool_name = match.group("name").lower()
        tool_body = match.group("body").strip()
    else:
        prose = raw.strip()
        tool_name = ""
        tool_body = ""
    return {"prose": prose, "tool_name": tool_name, "tool_body": tool_body}


def _iter_run(entry: dict) -> Iterator[dict]:
    """
    Generator yielding typed SSE event dicts for one eval entry.

    Runs synchronously — intended to be called from a thread pool.
    """
    scientist = ScientistAgent(api_url=SCIENTIST_API_URL)
    simulator = SimulatorAgent(model=SIMULATOR_MODEL)
    judge = JudgeAgent(model=JUDGE_MODEL)
    router = ToolRouter(simulator=simulator, max_turns=entry.get("max_turns", 10))

    scientist.add_user_message(entry["question"])
    yield {"type": "question", "content": entry["question"]}

    num_experiments = 0
    final_hypothesis = ""
    final_confidence = "medium"

    while True:
        raw = scientist.generate()
        parsed = _parse_scientist_output(raw)
        yield {"type": "scientist", **parsed}

        result = router.dispatch(raw)

        if result.action == LoopAction.CONTINUE:
            num_experiments += 1
            payload_str = str(result.payload)
            success = "Difficulty encountered" not in payload_str
            yield {
                "type": "simulator",
                "experiment_number": num_experiments,
                "success": success,
                "content": payload_str,
            }
            scientist.add_user_message(payload_str)

        elif result.action == LoopAction.STOP:
            if isinstance(result.payload, ConcludeArgs):
                final_hypothesis = result.payload.hypothesis
                final_confidence = result.payload.confidence.value
            else:
                # Timeout — treat entire last generation as the hypothesis
                final_hypothesis = raw.strip()
                final_confidence = "low"
            yield {
                "type": "conclude",
                "hypothesis": final_hypothesis,
                "confidence": final_confidence,
            }
            break

        elif result.action == LoopAction.REPROMPT:
            scientist.add_user_message(str(result.payload))
            yield {"type": "reprompt", "message": str(result.payload)}

    score = judge.score(
        question=entry["question"],
        hypothesis=final_hypothesis,
        ground_truth=entry["ground_truth"],
        key_concepts=entry.get("key_concepts", []),
        num_experiments=num_experiments,
        max_turns=entry.get("max_turns", 10),
    )
    yield {
        "type": "judge",
        "score": {
            "phenomenon": int(score.phenomenon_identified),
            "mechanism": int(score.qualitative_mechanism),
            "quantitative": int(score.quantitative_form),
            "confidence_calibrated": int(score.confidence_calibrated),
            "total": score.total,
            "reasoning": score.reasoning,
        },
    }


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "scenarios": len(_EVAL_SET),
        "scientist_api_url": SCIENTIST_API_URL,
        "simulator_model": SIMULATOR_MODEL,
        "judge_model": JUDGE_MODEL,
    })


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


@app.get("/api/run/{scenario_id}")
async def run_scenario(scenario_id: str, request: Request) -> StreamingResponse:
    """Stream a live experiment run as Server-Sent Events."""
    entry = _EVAL_INDEX.get(scenario_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Scenario '{scenario_id}' not found. Available: {list(_EVAL_INDEX)}",
        )

    event_queue: queue.Queue = queue.Queue()

    def pipeline_thread() -> None:
        try:
            for event in _iter_run(entry):
                event_queue.put(event)
        except Exception as exc:
            event_queue.put({"type": "error", "message": str(exc)})
        finally:
            event_queue.put(None)  # sentinel

    _executor.submit(pipeline_thread)

    async def generate():
        while True:
            if await request.is_disconnected():
                break
            try:
                event = event_queue.get_nowait()
            except queue.Empty:
                import asyncio
                await asyncio.sleep(0.1)
                continue
            if event is None:
                yield "data: {\"type\": \"done\"}\n\n"
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
