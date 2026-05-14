# results/

Output directory for all eval run results.

## Structure

```
results/
└── runs/
    └── <run_id>/          # e.g. 20260514T143021_a3f2bc/
        ├── summary.json   # Aggregated scores for all entries in the run
        ├── free_fall_galileo.jsonl
        ├── photoelectric_effect.jsonl
        ├── uv_catastrophe.jsonl
        └── special_relativity_michelson_morley.jsonl
```

Run IDs are UTC timestamps + 6-character hex suffix (e.g. `20260514T143021_a3f2bc`).

## Per-Entry JSONL Format

Each `.jsonl` file contains a single JSON object with the following fields:

```json
{
  "entry_id": "photoelectric_effect",
  "tier": "target",
  "domain": "optics / electromagnetism",
  "difficulty": 3,
  "num_experiments": 7,
  "max_turns": 10,
  "final_hypothesis": "The scientist's final conclude() text...",
  "score": {
    "total": 3,
    "phenomenon_identified": 1,
    "qualitative_mechanism": 1,
    "quantitative_form": 1,
    "confidence_calibrated": 0,
    "reasoning": "Judge's free-text reasoning per dimension..."
  },
  "turn_history": [
    { "role": "user", "content": "Opening question..." },
    { "role": "assistant", "content": "TOOL: propose_experiment\n..." },
    { "role": "user", "content": "Laboratory report: Sir, I have carried out..." },
    "..."
  ]
}
```

## Summary Format

`summary.json` contains the run ID and the full list of per-entry records,
suitable for loading into a notebook for analysis.

## Analysis

To aggregate results across runs or plot convergence, load `summary.json`
files from each run directory. Key metrics to track:

- `score.total` per tier (baseline / target / stretch)
- `num_experiments` relative to `max_turns` (efficiency)
- `score.reward_hacking_flag` rate across entries (see [Concern #5](../README.md#5-judge-reward-hacking))
- Dimension breakdown: which of the four rubric dimensions are systematically failed
