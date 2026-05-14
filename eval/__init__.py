"""
eval/
-----
Evaluation set definitions and the main orchestration loop.

- eval_set.json : curated (question, ground_truth, key_concepts) entries
- runner        : loads the eval set, runs the full experiment loop per entry,
                  and writes per-run results to results/runs/
"""
