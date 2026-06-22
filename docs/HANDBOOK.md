# Legal Document Summarizer & Clause Analyzer Handbook

This handbook explains the project end-to-end using the actual implemented code and real executed artifacts in this repository.

It is designed for:

- beginners learning legal AI engineering from scratch
- intermediate/advanced engineers who need implementation-level detail
- teams that want a reusable, local, auditable legal analysis component

## Table of Contents

1. Problem and objectives
2. Repository architecture and workflow
3. Data pipeline and supervision design
4. Baselines and prompting systems
5. Fine-tuning system (TRL + PEFT + optional Unsloth)
6. Evaluation framework and judge system
7. Error analysis and observed failure modes
8. Inference pipeline and production usage
9. Strict run validation and reproducibility
10. Real results interpretation and practical takeaways
11. References and appendix

## 1) Problem and Objectives

### Problem statement

Legal text understanding is difficult because obligations, liabilities, restrictions, and risk are often encoded in long, nested, cross-referenced language. Generic summarization is insufficient when the output must be legally structured and operationally useful.

### Project objective

Build a local system that outputs structured legal analysis for legal text:

- plain-English summary
- executive bullets
- clause-level explanations
- risk classification
- liability extraction
- obligation/right/restriction extraction
- red-flag detection
- recommended review areas

### Output contract in code

Defined in `src/legal_clause_analyzer/schemas.py` as `LegalAnalysis`.

## 2) Repository Architecture and Workflow

### What is this technique?

A staged legal AI pipeline with explicit separation of data preparation, baseline benchmarking, fine-tuning, evaluation, and inference.

### Why is it used?

Staging isolates failure points and gives auditable before-vs-after evidence (baseline vs fine-tuned).

### How it appears in the code

- orchestrator: `scripts/run_all.py`
- strict validator: `scripts/validate_strict_run.py`
- architecture note: `docs/ARCHITECTURE.md`

### Practical workflow (actual run order)

1. `scripts/prepare_data.py`
2. `scripts/run_baselines.py`
3. `scripts/train_model.py`
4. `scripts/evaluate.py`
5. `scripts/run_inference.py`
6. `scripts/generate_notebooks.py`
7. `scripts/execute_notebooks.py`
8. `scripts/validate_strict_run.py`

## 3) Data Pipeline and Supervision Design

### What is this technique?

Deterministic policy-mapped supervision from real legal datasets.

### Why is it used?

LexGLUE provides real legal labels, but not a single ready-made schema for executive summary + risk + liabilities + obligations. Deterministic mapping creates unified training targets without synthetic dataset replacement.

### How it appears in the code

- data ingestion and split preservation: `src/legal_clause_analyzer/data/lexglue_loader.py`
- policy mapping rules: `src/legal_clause_analyzer/data/policy_mapping.py`
- row transformation to supervised schema: `src/legal_clause_analyzer/data/instruction_builder.py`
- split write + dataset report: `scripts/prepare_data.py`

### Practical details from real artifacts

Source: `data/processed/dataset_report.json`

- train rows: `1540`
- validation rows: `97`
- test rows: `337`

Included LexGLUE subsets:

- `ledgar`
- `unfair_tos`
- `scotus`
- `eurlex`
- `case_hold`

Current dataset limitations (artifact-backed):

- no native executive-summary labels across all tasks
- no direct gold risk/liability taxonomy for the unified output schema
- cross-jurisdiction semantics remain challenging for compact local models

## 4) Baselines and Prompting Systems

### What is this technique?

Baseline benchmarking with prompt variants before any fine-tuning.

### Why is it used?

Without baselines, improvement claims are not meaningful.

### How it appears in the code

- prompt templates and JSON parser fallback: `src/legal_clause_analyzer/baselines/generation.py`
- model clients (`ollama`, `transformers`): `src/legal_clause_analyzer/baselines/model_clients.py`
- baseline runner: `scripts/run_baselines.py`

### Baselines implemented

- `prompt_only`
- `few_shot`
- `granite_zero_shot`
- `qwen_zero_shot`

### Practical note from predictions

Current baseline prediction files are in `artifacts/runs/balanced_local_v1/predictions_*.jsonl`.

In this run, all baseline systems produced similar top-line metrics (see section 10).

## 5) Fine-Tuning System (TRL + PEFT + optional Unsloth)

### What is this technique?

Parameter-efficient supervised fine-tuning (LoRA/QLoRA) on legal instruction-response rows.

### Why is it used?

Full-model fine-tuning is usually impractical on local 8GB-class GPUs. QLoRA enables adapter tuning with lower VRAM footprint.

### How it appears in the code

- training implementation: `src/legal_clause_analyzer/finetune/train.py`
- SFT text format: `src/legal_clause_analyzer/finetune/dataset_format.py`
- training entrypoint and GPU/runtime report: `scripts/train_model.py`

### TRL

- definition: training framework for LLM fine-tuning with `SFTTrainer`
- where used: `SFTConfig`, `SFTTrainer` in `train.py`
- what changed: structured trainer/eval/checkpoint flow instead of ad-hoc loops

### PEFT

- definition: adapter-based parameter-efficient tuning
- where used: `prepare_model_for_kbit_training`, `LoraConfig`, `get_peft_model`
- what changed: feasible 4-bit adapter training on local hardware

### Unsloth

- definition: optional acceleration backend for LoRA/QLoRA workflows
- where used: optional `_load_unsloth_bundle` path in `train.py`
- what changed in this environment: attempted first, not available, clean fallback to TRL+PEFT

Runtime proof: `artifacts/models/finetuned_adapter/training_backend_report.json`

- `backend_order`: `['unsloth', 'trl']`
- `selected_backend`: `trl_peft`
- `selected_model`: `ibm-granite/granite-4.1-3b`

### Practical resource/runtime snapshot

Source: `artifacts/metrics/training_runtime_report.json`

- `cuda_available`: `true`
- elapsed: `146.639812707901` seconds
- max CUDA allocated: `5549631488` bytes
- max CUDA reserved: `5775556608` bytes

## 6) Evaluation Framework and Judge System

### What is this technique?

Multi-axis evaluation combining automatic metrics, extraction/classification scores, hallucination diagnostics, latency, and LLM-as-a-judge.

### Why is it used?

Legal systems need both text quality and structured legal utility. A single metric is insufficient.

### How it appears in the code

- metric computation: `src/legal_clause_analyzer/evaluation/metrics.py`
- hallucination diagnostics: `src/legal_clause_analyzer/evaluation/hallucination.py`
- judge logic: `src/legal_clause_analyzer/evaluation/judge.py`
- plotting: `src/legal_clause_analyzer/evaluation/plots.py`
- evaluator orchestration: `scripts/evaluate.py`

### Metrics implemented

Summarization:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- BLEU
- METEOR
- BERTScore

Classification / extraction:

- risk: accuracy, precision, recall, F1
- obligation extraction: precision, recall, F1
- liability extraction: precision, recall, F1
- red-flag extraction: precision, recall, F1

Judge dimensions:

- legal accuracy
- summary quality
- risk identification quality
- clause extraction quality
- hallucination risk
- executive usefulness
- overall

## 7) Error Analysis and Observed Failure Modes

### What is this technique?

Artifact-driven post-run failure analysis.

### Why is it used?

To move from "did metrics go up" to "what is still broken and why".

### How it appears in code and artifacts

- hallucination diagnostics code: `evaluation/hallucination.py`
- reports: `artifacts/metrics/hallucination_report.json`
- confusion and score plots: `artifacts/figures/risk_confusion_matrix.png`, `artifacts/figures/system_scoreboard.png`

### Real observed failure pattern in this run

- `unsupported_claim_rate` is low (`0.0`), but
- `missing_obligation_rate` and `missing_liability_rate` remain `1.0` across systems

Interpretation: strict unsupported claim control is not translating into recall for obligation/liability extraction.

## 8) Inference Pipeline and Production Usage

### What is this technique?

Adapter-first inference with controlled fallback behavior.

### Why is it used?

Provides a reliable production path:

- use fine-tuned adapter when available
- keep strict mode for validation workflows
- permit fallback behavior in non-strict runtime scenarios

### How it appears in the code

- main inference logic: `src/legal_clause_analyzer/inference/pipeline.py`
- inference script: `scripts/run_inference.py`

### Practical behavior from current artifacts

Source: `artifacts/inference/inference_examples.json`

- strategy used: `fine_tuned_adapter` for all examples
- model used: `ibm-granite/granite-4.1-3b`
- output content shows parser-fallback style values in examples (`"No summary generated."` + mostly empty structured fields)

This indicates runtime path correctness but quality bottlenecks in generation/parsing for these examples.

## 9) Strict Run Validation and Reproducibility

### What is this technique?

Post-run integrity checks to ensure outputs are real, complete, and fallback-free in strict mode.

### Why is it used?

Legal AI claims require auditable run integrity.

### How it appears in the code

`scripts/validate_strict_run.py` verifies:

- required predictions exist
- no fallback summaries in predictions
- judge report entries exist and are non-empty
- CUDA availability in training runtime report
- inference strategies are fine-tuned adapter only
- executed notebooks exist (when required)

## 10) Real Results Interpretation and Practical Takeaways

### Key results (artifact-backed)

Sources:

- `artifacts/metrics/system_metrics.json`
- `artifacts/metrics/scoreboard.csv`
- `artifacts/metrics/judge_report.json`
- `artifacts/metrics/latency_report.json`

Fine-tuned adapter vs baseline (`few_shot`):

- ROUGE-L: `0.0253431362` vs `0.0023148148` (`+0.0230283214`)
- BERTScore F1: `0.8351356983` vs `0.8335364461` (`+0.0015992522`)
- Risk F1: `0.1502192982` vs `0.1085271318` (`+0.0416921665`)
- Obligation F1: tie at `0.25`
- Liability F1: tie at `0.0`
- Red-Flag F1: tie at `1.0`

Judge mean overall:

- granite judge: `2.7777777778` (fine-tuned) vs `2.7222222222` (baseline)
- qwen judge: `3.0` vs `3.0` (tie)

Latency mean (seconds):

- fine-tuned adapter: `7.9773901738`
- baseline systems: `1.5934` to `2.3928`

### Practical takeaway

Fine-tuning improves summary and risk signals in this run, but liability and obligation extraction recall remain the main technical gap. The system is functionally complete and reproducible, but not yet strong enough to skip downstream legal review.

## 11) References and Appendix

### Official references used in implementation decisions

- TRL SFT docs: https://huggingface.co/docs/trl/sft_trainer
- PEFT quantization guide: https://huggingface.co/docs/peft/main/developer_guides/quantization
- Unsloth docs: https://docs.unsloth.ai/
- LexGLUE dataset card: https://huggingface.co/datasets/coastalcph/lex_glue

### Related project files

- config: `configs/default.yaml`
- architecture summary: `docs/ARCHITECTURE.md`
- notebook sequence: `notebooks/01_*` through `notebooks/07_*`

### Legal disclaimer

This project is a technical implementation and educational resource. It is not legal advice.
