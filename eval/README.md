# eval/

The evaluation set and orchestration loop.

## Files

| File | Purpose |
|---|---|
| `eval_set.json` | Curated (question, ground_truth, key_concepts) entries |
| `runner.py` | Loads eval set, runs the experiment loop, writes results |

## Eval Set Design

Each entry in `eval_set.json` covers a phenomenon that:
1. Was **not** well-understood before the training cutoff (Jan 1 1900), reducing contamination risk
2. Can be meaningfully investigated through experiments a 19th-century laboratory could plausibly run
3. Has a clear, checkable ground truth

### Difficulty tiers

| Tier | Description | Examples |
|---|---|---|
| `baseline` | Well within pre-1900 knowledge; should be trivially solved. Validates the pipeline. | Galileo's free-fall, Hooke's law |
| `target` | Discovered shortly after 1900; the core challenge. | Photoelectric effect, UV catastrophe |
| `stretch` | Requires genuine conceptual leap. | Special relativity, equivalence principle |

### Contamination risk field

Each entry includes a `pre1900_knowledge` field documenting what a well-read
Victorian scientist would already know about this domain. This is used to
assess whether a correct answer is likely genuine reasoning or retrieval.

See [Concern #6 (Eval Set Contamination)](../README.md#6-eval-set-contamination).

## Runner

`runner.py` orchestrates a single full eval run:

1. Load `eval_set.json`
2. For each entry:
   a. Reset all agents and the tool router
   b. Inject the question into the scientist's context
   c. Run the experiment loop (scientist → router → simulator → scientist …)
   d. On `conclude` or max turns: pass hypothesis to the judge
   e. Write the full turn history + score to `results/runs/<run_id>/<entry_id>.jsonl`
3. Print a summary table of scores

### Usage

Requires the GCP SSH tunnel to be open (`SCIENTIST_API_URL=http://localhost:8000`) and `OPEN_ROUTER_API_KEY` set in `.env`.

```bash
python eval/runner.py \
  --eval eval/eval_set.json \
  --out results/runs/ \
  --scientist-url http://localhost:8000 \
  --judge-model anthropic/claude-3.5-haiku \
  --simulator-model anthropic/claude-3.5-haiku \
  --noise-level 0.05 \
  --failure-rate 0.10
```
