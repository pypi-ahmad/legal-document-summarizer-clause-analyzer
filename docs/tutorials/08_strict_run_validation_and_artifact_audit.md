# Tutorial 08: Strict Run Validation and Artifact Audit

## Learning Goals

- understand strict run guarantees in this project
- know exactly what is verified post-run
- use artifact checks for reproducibility and auditability

## What Is This Technique?

### Definition

Strict-run validation is a post-execution integrity gate that verifies artifact completeness and disallows hidden fallback behavior.

### Why It Is Used

In legal workflows, reproducibility and evidence integrity are as important as raw performance.

### How It Appears in the Code

- validator script: `scripts/validate_strict_run.py`
- strict flags in orchestrators:
  - `scripts/run_all.py`
  - `scripts/run_baselines.py`
  - `scripts/train_model.py`
  - `scripts/evaluate.py`
  - `scripts/run_inference.py`

### Practical Explanation in This Project

The strict validator checks:

- all prediction files exist
- zero fallback rows in predictions
- judge report entries are non-empty
- CUDA availability recorded in training runtime
- inference strategy is `fine_tuned_adapter`
- executed notebook outputs exist (when required)

## Real Artifact Audit Checklist

Core reports:

- `artifacts/metrics/system_metrics.json`
- `artifacts/metrics/judge_report.json`
- `artifacts/metrics/latency_report.json`
- `artifacts/metrics/hallucination_report.json`
- `artifacts/metrics/training_runtime_report.json`

Predictions:

- `artifacts/runs/balanced_local_v1/predictions_*.jsonl`

Inference examples:

- `artifacts/inference/inference_examples.json`

Notebook execution outputs:

- `notebooks/*.executed.ipynb`

## Interpreting a Passing Strict Run

A strict pass means:

- the workflow completed end-to-end
- expected outputs exist and are structurally valid
- no hidden deterministic fallback rows were accepted in strict prediction paths

It does not mean legal quality is solved. Quality still depends on the metric profile and error analysis outcomes.

## Key Takeaway

Strict validation provides trust in run integrity; it is the foundation for credible comparison and iterative model improvement.
