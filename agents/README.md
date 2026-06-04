# agents/

LLM wrappers for the three roles in the pipeline. Each agent is responsible for one role only — they do not call each other directly; the orchestrator (`eval/runner.py`) coordinates them.

## Files

| File | Role | Backing model |
|---|---|---|
| `scientist.py` | ScientistAgent | GPT-1900 (self-hosted on GCP, NVIDIA L4 via SSH tunnel) |
| `simulator.py` | SimulatorAgent | OpenRouter (default: `anthropic/claude-3.5-haiku`) |
| `judge.py` | JudgeAgent | OpenRouter (default: `anthropic/claude-3.5-haiku`) |
| `llm_client.py` | shared HTTP client | — |

## LLM Client (`llm_client.py`)

Both the simulator and judge call OpenRouter through a single shared `call_llm(model, prompt)` function in `llm_client.py`. OpenRouter exposes an OpenAI-compatible API, so only `httpx` is needed.

Required environment variable:
```
OPEN_ROUTER_API_KEY=sk-or-v1-...
```

The default model is `anthropic/claude-3.5-haiku`. Any model from the [OpenRouter catalogue](https://openrouter.ai/models) can be used by setting `SIMULATOR_MODEL` / `JUDGE_MODEL` in `.env`.

## Scientist Agent

The scientist drives the experiment loop. It receives the opening research question and writes natural language prose describing experiments it wants to run. The `ToolRouter` uses a modern LLM to classify each response and dispatch accordingly.

**Key design constraints:**
- Temperature `0.6`, `top_k=20` — tighter sampling to reduce degeneration
- GPT-1900 uses a chat template with special tokens (`<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`, `<|assistant_end|>`); `_build_prompt()` uses these instead of plain `USER:` / `ASSISTANT:` markers
- Special tokens are stripped from responses before storing in context, preventing format pollution across turns
- Repetition detection truncates degenerate "A. B. C." loops at the client
- Context is maintained across turns; older experiment results are compressed once context approaches the 2048-token limit
- The backing HTTP client points at `SCIENTIST_API_URL` (default `http://localhost:8000`) — keep the GCP SSH tunnel open (see `deploy/README.md`)

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
