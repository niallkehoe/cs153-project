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
| Scientist | GPT-1900 (DigitalOcean GPU) | Proposes experiments, interprets results, concludes |
| Simulator | Cloudflare Workers AI | Returns raw observational data with configurable noise |
| Judge | Cloudflare Workers AI | Scores final hypothesis against ground truth with partial credit |

---

## Directory Structure

```
cs153-project/
├── agents/         # LLM agent wrappers (scientist, simulator, judge)
├── tools/          # Tool schema definitions and call router
├── prompts/        # System prompt templates for each agent
├── eval/           # Eval set, orchestration loop, results
├── deploy/         # DigitalOcean GPU server setup for GPT-1900
└── results/        # Per-run JSONL output files
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set credentials
cp .env.example .env
# Fill in CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, and SCIENTIST_API_URL

# 3. Run eval against a hosted GPT-1900 instance
python eval/runner.py \
  --scientist-url http://<droplet-ip>:8000 \
  --eval eval/eval_set.json \
  --out results/runs/

# 4. (Optional) Override the Cloudflare model
python eval/runner.py \
  --scientist-url http://<droplet-ip>:8000 \
  --judge-model @cf/meta/llama-3.1-8b-instruct-fast \
  --simulator-model @cf/meta/llama-3.1-8b-instruct-fast

# 5. Spin up the GPT-1900 server on DigitalOcean — see deploy/README.md
python deploy/server.py --model-dir /path/to/gpt1900
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
