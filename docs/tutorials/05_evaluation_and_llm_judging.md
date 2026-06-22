# Tutorial 05: Evaluation and LLM-as-a-Judge

## Learning Goals

- understand every metric family used
- understand how judge scoring is implemented
- connect metrics to real artifact outputs

## What Is This Technique?

### Definition

A multi-dimensional evaluation stack combining automatic metrics, extraction/classification metrics, hallucination diagnostics, latency, and LLM-as-a-judge scoring.

### Why It Is Used

Legal AI quality is multi-objective. Good summary text alone does not imply good risk or liability extraction.

### How It Appears in the Code

- automatic metrics: `src/legal_clause_analyzer/evaluation/metrics.py`
- judge prompt and parsing: `src/legal_clause_analyzer/evaluation/judge.py`
- hallucination diagnostics: `src/legal_clause_analyzer/evaluation/hallucination.py`
- evaluation orchestration: `scripts/evaluate.py`

### Practical Explanation in This Project

`evaluate.py` reads all prediction files, computes metrics, runs two judge models, writes report JSON files, and creates visualization figures.

## Metric Families

Summarization:

- ROUGE-1/2/L
- BLEU
- METEOR
- BERTScore

Classification/extraction:

- risk accuracy/precision/recall/F1
- obligation precision/recall/F1
- liability precision/recall/F1
- red-flag precision/recall/F1

Judge dimensions:

- legal accuracy
- summary quality
- risk identification quality
- clause extraction quality
- hallucination risk
- executive usefulness
- overall

## Real Results (Fine-Tuned vs Baseline)

From `artifacts/metrics/system_metrics.json` and `scoreboard.csv`:

- ROUGE-L: `0.0253431362` vs `0.0023148148`
- BERTScore F1: `0.8351356983` vs `0.8335364461`
- Risk F1: `0.1502192982` vs `0.1085271318`

From `artifacts/metrics/judge_report.json`:

- granite judge overall: `2.7777777778` vs `2.7222222222`
- qwen judge overall: tie at `3.0`

## Latency View

From `artifacts/metrics/latency_report.json`:

- fine-tuned mean latency: `7.9773901738s`
- baseline means: `1.5934s` to `2.3928s`

This highlights the speed-quality tradeoff for adapter inference.

## Key Takeaway

Evaluation must be interpreted as a bundle. In this run, summary and risk improved, but extraction recall remains the dominant limitation.
