# Tutorial 03: Baselines and Prompting Systems

## Learning Goals

- understand baseline systems and prompt styles
- understand why baseline-first evaluation is mandatory
- map prompt behavior to current metrics

## What Is This Technique?

### Definition

Baseline benchmarking is the controlled evaluation of non-fine-tuned systems before adaptation.

### Why It Is Used

It defines the reference point for all improvement claims.

### How It Appears in the Code

- baseline specs and prompt construction: `src/legal_clause_analyzer/baselines/generation.py`
- generation backends and parsing: `src/legal_clause_analyzer/baselines/model_clients.py`
- baseline execution script: `scripts/run_baselines.py`

### Practical Explanation in This Project

The project runs four baseline systems on held-out test rows and writes per-system `predictions_*.jsonl` files with latency measurements.

## Baseline Systems Implemented

- `prompt_only`
- `few_shot`
- `granite_zero_shot`
- `qwen_zero_shot`

Files:

- `artifacts/runs/balanced_local_v1/predictions_prompt_only.jsonl`
- `artifacts/runs/balanced_local_v1/predictions_few_shot.jsonl`
- `artifacts/runs/balanced_local_v1/predictions_granite_zero_shot.jsonl`
- `artifacts/runs/balanced_local_v1/predictions_qwen_zero_shot.jsonl`

## Prompt and Parsing Strategy

### What Is This Technique?

Schema-constrained prompting + best-effort JSON parsing.

### Why It Is Used

Keeps output close to the legal schema and reduces free-form drift.

### How It Appears in the Code

- system prompt requires JSON keys: `SYSTEM_PROMPT` in `generation.py`
- parser fallback logic: `try_parse_json` in `model_clients.py`
- safe conversion to `LegalAnalysis`: `safe_analysis_from_text` in `generation.py`

### Practical Observation

Current predictions include many parser-fallback style outputs (`"No summary generated."`), which explains weak extraction recall despite stable run completion.

## Real Baseline Performance Snapshot

From `artifacts/metrics/system_metrics.json`:

- baseline ROUGE-L around `0.0023148148`
- baseline risk F1 around `0.1085271318`

These are the values the fine-tuned model is compared against.

## Key Takeaway

Baselines reveal that local prompt-only/legal-prompt systems can run reliably but produce limited legal extraction quality without adaptation.
