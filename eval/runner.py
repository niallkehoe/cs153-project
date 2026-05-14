"""
runner.py
---------
Main orchestration loop for the re-discovery eval pipeline.

Loads the eval set, runs the full experiment loop for each entry, and writes
per-run results to results/runs/<run_id>/.

Loop structure per eval entry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  1. Reset ScientistAgent, ToolRouter
  2. Inject the opening question as the first user message to the scientist
  3. Loop:
       a. scientist.generate()  → raw generation text
       b. router.dispatch(text) → RouterResult
       c. if CONTINUE: add result to scientist context; repeat
       d. if STOP:     pass hypothesis + metadata to judge; break
       e. if REPROMPT: add re-prompt message to scientist context; repeat
  4. judge.score(...)           → JudgeScore
  5. Write full turn history + score to JSONL

See CONCERNS in README.md:
  #3  Context Window Pressure  — managed by ScientistAgent._maybe_compress_history
  #8  Max Turns and Termination — enforced by ToolRouter

Usage
~~~~~
  python eval/runner.py \\
    --eval eval/eval_set.json \\
    --out results/runs/ \\
    --scientist-url http://<droplet-ip>:8000 \\
    --judge-model claude-sonnet-4-5 \\
    --simulator-model claude-sonnet-4-5 \\
    --noise-level 0.05 \\
    --failure-rate 0.10
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.judge import JudgeAgent, JudgeScore
from agents.scientist import ScientistAgent
from agents.simulator import SimulatorAgent
from tools.router import LoopAction, ToolRouter


def run_entry(
    entry: dict[str, Any],
    scientist: ScientistAgent,
    simulator: SimulatorAgent,
    judge: JudgeAgent,
    router: ToolRouter,
) -> dict[str, Any]:
    """
    Run the full experiment loop for a single eval entry.

    Parameters
    ----------
    entry:
        A single eval set entry loaded from eval_set.json.
    scientist:
        ScientistAgent instance (will be reset before use).
    simulator:
        SimulatorAgent instance.
    judge:
        JudgeAgent instance.
    router:
        ToolRouter instance (will be reset before use, max_turns set from entry).

    Returns
    -------
    dict
        Full run record: metadata, turn history, and judge score.
        Written to results/runs/<run_id>/<entry_id>.jsonl by the caller.
    """
    scientist.reset()
    router.reset()
    router.max_turns = entry.get("max_turns", 10)

    turn_history: list[dict[str, Any]] = []
    scientist.add_user_message(entry["question"])

    final_hypothesis: str = ""
    num_experiments = 0

    while True:
        raw = scientist.generate()
        turn_history.append({"role": "assistant", "content": raw})

        result = router.dispatch(raw)

        if result.action == LoopAction.CONTINUE:
            num_experiments += 1
            scientist.add_user_message(str(result.payload))
            turn_history.append({"role": "user", "content": str(result.payload)})

        elif result.action == LoopAction.STOP:
            if hasattr(result.payload, "hypothesis"):
                final_hypothesis = result.payload.hypothesis
            else:
                final_hypothesis = str(result.payload)
            break

        elif result.action == LoopAction.REPROMPT:
            scientist.add_user_message(str(result.payload))
            turn_history.append({"role": "user", "content": str(result.payload), "reprompt": True})

    score: JudgeScore = judge.score(
        question=entry["question"],
        hypothesis=final_hypothesis,
        ground_truth=entry["ground_truth"],
        key_concepts=entry["key_concepts"],
        num_experiments=num_experiments,
        max_turns=entry.get("max_turns", 10),
    )

    return {
        "entry_id": entry["id"],
        "tier": entry.get("tier"),
        "domain": entry.get("domain"),
        "difficulty": entry.get("difficulty"),
        "num_experiments": num_experiments,
        "max_turns": entry.get("max_turns", 10),
        "final_hypothesis": final_hypothesis,
        "score": {
            "total": score.total,
            "phenomenon_identified": score.phenomenon_identified,
            "qualitative_mechanism": score.qualitative_mechanism,
            "quantitative_form": score.quantitative_form,
            "confidence_calibrated": score.confidence_calibrated,
            "reasoning": score.reasoning,
        },
        "turn_history": turn_history,
    }


def main() -> None:
    """
    CLI entry point.

    Parses arguments, instantiates agents, iterates over the eval set,
    and writes results to the output directory.
    """
    parser = argparse.ArgumentParser(description="Run the re-discovery eval pipeline.")
    parser.add_argument("--eval", default="eval/eval_set.json", help="Path to eval set JSON")
    parser.add_argument("--out", default="results/runs/", help="Output directory for run results")
    parser.add_argument("--scientist-url", required=True, help="GPT-1900 server URL")
    parser.add_argument(
        "--judge-model",
        default="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        help="Cloudflare Workers AI model ID for the judge",
    )
    parser.add_argument(
        "--simulator-model",
        default="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        help="Cloudflare Workers AI model ID for the simulator",
    )
    parser.add_argument("--noise-level", type=float, default=0.05, help="Simulator noise level [0,1]")
    parser.add_argument("--failure-rate", type=float, default=0.10, help="Simulator failure rate [0,1]")
    parser.add_argument("--ids", nargs="*", help="Run only these eval entry IDs (default: all)")
    args = parser.parse_args()

    with open(args.eval) as f:
        eval_set = json.load(f)

    if args.ids:
        eval_set = [e for e in eval_set if e["id"] in args.ids]

    scientist = ScientistAgent(api_url=args.scientist_url)
    simulator = SimulatorAgent(model=args.simulator_model, noise_level=args.noise_level, failure_rate=args.failure_rate)
    judge = JudgeAgent(model=args.judge_model)
    router = ToolRouter(simulator=simulator)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run ID: {run_id}")
    print(f"Output: {out_dir}")
    print(f"Entries: {len(eval_set)}\n")

    scores: list[dict[str, Any]] = []
    for entry in eval_set:
        print(f"  [{entry['id']}] running...")
        record = run_entry(entry, scientist, simulator, judge, router)
        scores.append(record)

        out_file = out_dir / f"{entry['id']}.jsonl"
        with open(out_file, "w") as f:
            f.write(json.dumps(record) + "\n")

        print(f"  [{entry['id']}] score: {record['score']['total']}/4  "
              f"experiments: {record['num_experiments']}")

    # Summary
    print(f"\n{'Entry':<40} {'Score':>6} {'Exps':>6}")
    print("-" * 55)
    for s in scores:
        print(f"  {s['entry_id']:<38} {s['score']['total']:>5}/4 {s['num_experiments']:>5}")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"run_id": run_id, "results": scores}, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
