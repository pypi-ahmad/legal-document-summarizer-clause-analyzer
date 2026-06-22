# Tutorial 06: Error Analysis and Visualization

## Learning Goals

- identify concrete failure patterns from current artifacts
- interpret visual diagnostics correctly
- map failures back to implementation constraints

## What Is This Technique?

### Definition

Structured error analysis is the process of converting raw predictions and metrics into actionable model improvement hypotheses.

### Why It Is Used

Without error analysis, you only know whether metrics changed, not what to fix next.

### How It Appears in the Code

- hallucination metrics logic: `src/legal_clause_analyzer/evaluation/hallucination.py`
- plotting functions: `src/legal_clause_analyzer/evaluation/plots.py`
- metric/judge aggregation for analysis: `scripts/evaluate.py`

### Practical Explanation in This Project

The evaluation stage writes machine-readable reports and figures to `artifacts/metrics/` and `artifacts/figures/`, enabling repeatable post-run diagnosis.

## Real Failure Signals in This Run

From `artifacts/metrics/hallucination_report.json`:

- unsupported claim rate: `0.0`
- missing obligation rate: `1.0`
- missing liability rate: `1.0`

Interpretation:

- the system avoids adding unsupported extraction items
- but frequently fails to recover required obligation/liability items

## Visualization Assets

- `artifacts/figures/training_curve.png`
- `artifacts/figures/metric_bars.png`
- `artifacts/figures/system_scoreboard.png`
- `artifacts/figures/risk_confusion_matrix.png`
- `artifacts/figures/dataset_distributions.png`

## Root-Cause Hypotheses (Implementation-Consistent)

1. supervision mapping provides conservative signals but limited fine-grained liability labels
2. structured parser fallback (`safe_analysis_from_text`) can collapse outputs into empty fields
3. compact model size + short sequence constraints limit nuanced extraction retention

## Key Takeaway

The next gains are likely to come from better supervision richness and decoding/parsing robustness, not only from repeating the current training recipe.
