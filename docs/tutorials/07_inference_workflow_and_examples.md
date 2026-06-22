# Tutorial 07: Inference Workflow and Example Outputs

## Learning Goals

- understand adapter-first inference behavior
- understand strict vs non-strict inference behavior
- interpret current example outputs correctly

## What Is This Technique?

### Definition

Adapter-first inference tries the fine-tuned adapter path first, then uses fallback paths depending on strictness and runtime conditions.

### Why It Is Used

Balances production reliability (fallbacks) with strict validation integrity (fail-fast when required).

### How It Appears in the Code

- inference orchestration: `src/legal_clause_analyzer/inference/pipeline.py`
- runtime script for examples: `scripts/run_inference.py`
- strict behavior tests: `tests/test_strict_inference.py`

### Practical Explanation in This Project

The script runs three legal examples and writes `artifacts/inference/inference_examples.json`, including model used, strategy, latency, and full structured output.

## Current Real Example Outcomes

Source: `artifacts/inference/inference_examples.json`

- strategy for all three examples: `fine_tuned_adapter`
- model for all three examples: `ibm-granite/granite-4.1-3b`
- outputs show parser fallback style values:
  - `executive_summary`: `"No summary generated."`
  - empty extraction arrays
  - default risk rationale fallback text

## Why This Matters

Pipeline routing is functioning correctly (adapter path selected), but output quality on these examples indicates unresolved generation/parsing robustness issues.

## Production Integration Notes

The reusable function is `analyze_legal_text` in `inference/pipeline.py`.

`InferenceResult` returns:

- `analysis` (structured `LegalAnalysis`)
- `model_used`
- `strategy`

This is directly usable as a backend component in larger legal AI systems.

## Key Takeaway

System-level integration is complete and strict-path compatible, but model output quality still requires targeted improvement work.
