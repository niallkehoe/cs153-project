# CS153 — Re-Discovery Eval Pipeline

An experiment in machine scientific reasoning: can a historically-constrained LLM (GPT-1900, trained exclusively on pre-1900 text) actively re-discover scientific principles by proposing and interpreting experiments — the way a real scientist would?

Inspired by Michael Hla's [Machina Mirabilis](https://michaelhla.com/blog/machina-mirabilis.html) and Demis Hassabis's proposed benchmark:

> _Pretrain an LLM on all text before a scientific breakthrough, prompt it with experimental observations, and ask it to explain the results._

This project extends that setup with **active experimentation**: rather than receiving pre-packaged observations, the scientist agent must decide what to test, receive simulated results, iterate, and converge on a conclusion — modelling the actual structure of scientific inquiry.

---

## Architecture

```
EvalSet (question + ground_truth)
        |
        v
   Orchestrator (eval/runner.py)
        |
        v
 Scientist Agent  <---[propose_experiment(params)]---> Tool Router
  (GPT-1900)                                               |
        ^                                                  v
        |                                      Simulator Agent (Modern LLM)
        +----------[raw observations + noise]--+          |
        |                                                  |
        +----[conclude(hypothesis)]-----------> Tool Router
                                                           |
                                                           v
                                                   Judge Agent (Modern LLM)
                                                           |
                                                           v
                                              Results (score + run metadata)
```

### Agents

| Agent | Model | Role |
|---|---|---|
| Scientist | GPT-1900 (GCP GPU — NVIDIA L4 via SSH tunnel) | Proposes experiments in natural language, interprets results, concludes |
| Simulator | OpenRouter (default: `anthropic/claude-3.5-haiku`) | Returns raw observational data with configurable noise |
| Judge | OpenRouter (default: `anthropic/claude-3.5-haiku`) | Scores final hypothesis against ground truth with partial credit |

---

## Directory Structure

```
cs153-project/
├── agents/         # LLM agent wrappers (scientist, simulator, judge)
├── tools/          # Tool schema definitions and call router
├── prompts/        # System prompt templates for each agent
├── eval/           # Eval set, orchestration loop, results
├── deploy/         # GCP GPU server setup for GPT-1900
├── web/            # FastAPI server + single-page frontend (live SSE pipeline)
└── results/        # Per-run JSONL output files
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -r web/requirements.txt

# 2. Set credentials
cp .env.example .env
# Fill in OPEN_ROUTER_API_KEY and SCIENTIST_API_URL

# 3. Open the SSH tunnel to the GCP GPU instance (keep this running)
gcloud compute ssh gpu-l4-1 --zone=us-central1-c --project=cs229-497921 -- -NL 8000:localhost:8000

# 4. Launch the web UI (separate terminal)
mamba activate sylvian
uvicorn web.server:app --reload --port 7860
# Then open http://127.0.0.1:7860

# 5. Or run the batch eval pipeline directly
python eval/runner.py \
  --scientist-url http://localhost:8000 \
  --eval eval/eval_set.json \
  --out results/runs/

# 6. Start the GPT-1900 server on the GCP VM (run on the VM)
python ~/cs153_server.py \
  --model-dir ~/gpt1900 \
  --model-files-dir ~/gpt1900_models/gpt1900-instruct-v3-sft \
  --host 127.0.0.1 --port 8000
```

---

## Concerns and Known Risks

These were identified during system design and should be tracked throughout development and evaluation.

### 1. Simulator Leakage
The modern LLM simulating experiments has full knowledge of post-1900 physics. There is a real risk it uses anachronistic framing (e.g., "photon", "quantum") or structures data in a way that reveals the correct theoretical interpretation. The simulator prompt must be tightly constrained to return **only raw observational data** — numbers, instrument readings, descriptions of phenomena — with zero theoretical interpretation. This should be tested adversarially.

### 2. Scientist Contamination (GPT-1900)
Hla documents that despite aggressive corpus filtering, some post-1900 leakage likely remains, and that the model has learned to "bullshit the judge" — producing plausible-sounding modern-theory phrases that the LLM judge rewards. The active experimentation loop makes pure pattern-matching harder (the model must commit to experimental parameters and receive consistent results), but does not eliminate this risk. Results should be treated as upper bounds.

### 3. Context Window Pressure
GPT-1900 has a 2048-token context window. A multi-turn experiment history (question + tool calls + results) will saturate this within 3–5 rounds. A compression/summarization strategy is needed — likely: compress older experiment results into a one-line summary after each turn, keeping only the most recent full result in context. This compression must itself not introduce post-1900 framing.

### 4. Binary Judge Scoring Is Insufficient
Pass/fail evaluation loses important signal. A scientist that correctly identifies *constant acceleration* but cannot name it as "gravitational acceleration" should score differently from one that concludes arbitrary force laws. The judge rubric should implement partial credit across: (a) correct phenomenon identified, (b) correct qualitative mechanism, (c) correct quantitative form, (d) appropriate confidence hedging.

### 5. Judge Reward Hacking
Hla observed the RL'd model latching onto phrases like "you have touched a point of considerable importance" to score well with the LLM judge. A similar failure mode can occur here: the scientist may learn (over many eval runs, if used for RL) to produce fluent-sounding Victorian prose that satisfies the judge without genuine mechanistic insight. Include a coherence check in the judge rubric and log raw generations for manual review.

### 6. Eval Set Contamination
Questions must be carefully chosen so the *phenomenon to rediscover* was not well-described in pre-1900 texts. Phenomena that had partial pre-1900 descriptions (e.g., Newton's corpuscular theory of light overlaps with photon arguments) are high contamination risk. Each eval entry should document the known pre-1900 state of knowledge in that domain.

### 7. Noise and Failure Mode Realism
Purely clean experimental data removes a key aspect of scientific reasoning — dealing with measurement error and confounders. The simulator should inject configurable Gaussian noise and occasionally return "experiment failed" results (broken equipment, confounding variable, etc.). Without this, the task is closer to data-fitting than scientific reasoning.

### 8. Max Turns and Termination
A scientist that converges in 2 experiments vs. one that meanders for 20 are meaningfully different. Max turns should be set per eval difficulty and tracked in results. The judge should also penalise conclusions reached only after excessive experimentation relative to the expected difficulty.

---

## Related Work

- [Machina Mirabilis — Michael Hla](https://michaelhla.com/blog/machina-mirabilis.html)
- [GPT-1900 GitHub](https://github.com/michaelhla/gpt1900)
- [GPT-1900 on HuggingFace](https://huggingface.co/mhla/gpt1900-d34-22btok)
- [TimeCapsuleLLM](https://github.com/haykgrigo3/TimeCapsuleLLM)
- [Hacker News discussion](https://news.ycombinator.com/item?id=46590280)

## AI Usage policy

I used AI coding tools to construct the harness for the agents (scientist, simulator, and judge).
I also used AI agents to build the frontend for the website.