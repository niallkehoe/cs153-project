# agents/

LLM wrappers for the three roles in the pipeline. Each agent is responsible for one role only — they do not call each other directly; the orchestrator (`eval/runner.py`) coordinates them.

## Files

| File | Role | Backing model |
|---|---|---|
| `scientist.py` | ScientistAgent | GPT-1900 (self-hosted on DigitalOcean GPU) |
| `simulator.py` | SimulatorAgent | Cloudflare Workers AI |
| `judge.py` | JudgeAgent | Cloudflare Workers AI |
| `llm_client.py` | shared HTTP client | — |

## LLM Client (`llm_client.py`)

Both the simulator and judge call Cloudflare Workers AI through a single shared `call_llm(model, prompt)` function in `llm_client.py`. This avoids vendor SDK dependencies — only `httpx` (already required for the scientist client) is needed.

Required environment variables:
```
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_API_TOKEN=...    # needs "Workers AI Run" permission
```

The default model is `@cf/meta/llama-3.3-70b-instruct-fp8-fast`. Any model from the [Cloudflare Workers AI catalogue](https://developers.cloudflare.com/workers-ai/models/) can be passed at runtime via `--judge-model` / `--simulator-model`.

## Scientist Agent

The scientist drives the experiment loop. It receives the opening research question and the tool schema, then iteratively generates either a `propose_experiment` tool call (requesting data) or a `conclude` tool call (ending the loop with a hypothesis).

**Key design constraints:**
- Temperature `0.7`, `top_k=50` — matches Hla's physics eval settings
- Context is maintained across turns; older experiment results are compressed to one-line summaries once context approaches the 2048-token limit
- The backing HTTP client points at the DigitalOcean `/generate` endpoint (see `deploy/server.py`)

## Simulator Agent

The simulator receives the scientist's proposed experiment and returns raw observational data. It has full modern physics knowledge but must be constrained by the system prompt (`prompts/simulator.txt`) to return **only** instrument readings, measurements, and factual observations — never theoretical interpretations, and never post-1900 vocabulary.

**Key design constraints:**
- Gaussian noise is injected at a configurable `noise_level` (default: 0.05, i.e. 5% relative error)
- A small `failure_rate` (default: 0.1) causes the simulator to return a plausible experiment failure instead of results, forcing the scientist to handle confounders
- See [Concern #1 (Simulator Leakage)](../README.md#1-simulator-leakage) and [Concern #7 (Noise Realism)](../README.md#7-noise-and-failure-mode-realism)

## Judge Agent

The judge evaluates the scientist's final `conclude()` call against the known ground truth from the eval set. It does not participate in the experiment loop — it only sees the final hypothesis and the ground truth.

**Scoring rubric (0–4):**
1. Correct phenomenon identified
2. Correct qualitative mechanism
3. Correct quantitative form (where applicable)
4. Appropriate confidence hedging (neither overconfident nor vacuous)

See [Concern #4 (Binary Scoring)](../README.md#4-binary-judge-scoring-is-insufficient) and [Concern #5 (Reward Hacking)](../README.md#5-judge-reward-hacking).
