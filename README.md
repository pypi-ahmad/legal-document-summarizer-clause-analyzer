# Legal Document Summarizer & Clause Analyzer

A local, production-style legal AI project that converts complex legal text into a structured analysis format.

This repository is tutorial-first and artifact-grounded: every reported result in this README comes from files currently present in `artifacts/` and `data/processed/`.

## Why This Exists

Legal text is hard for generic models because of long context dependencies, nested exceptions, cross-references, specialized terminology, and high hallucination risk. The project demonstrates a practical path from baseline prompting to domain-adapted fine-tuning for legal analysis workflows.

## What the System Produces

For each input legal text, the system returns a structured schema:

- `executive_summary`
- `executive_bullets`
- `clause_explanations`
- `risk_level` and `risk_rationale`
- `liabilities`
- `obligations`
- `rights`
- `restrictions`
- `red_flags`
- `recommended_review_areas`
- `evidence_spans`

Schema type: `LegalAnalysis` in `src/legal_clause_analyzer/schemas.py`.

## Real Pipeline (Code-Accurate)

End-to-end orchestration: `scripts/run_all.py`

1. `scripts/prepare_data.py`
2. `scripts/run_baselines.py`
3. `scripts/train_model.py`
4. `scripts/evaluate.py`
5. `scripts/run_inference.py`
6. `scripts/generate_notebooks.py`
7. `scripts/execute_notebooks.py`
8. `scripts/validate_strict_run.py` (strict mode)

Strict validation checks include:

- prediction files exist for all systems
- no fallback prediction rows
- non-empty judge metrics
- CUDA training confirmation
- inference strategy is `fine_tuned_adapter`
- executed notebook outputs exist

Validation script: `scripts/validate_strict_run.py`

## Model Roles and Rationale

Configured in `configs/default.yaml`.

- `qwen3-embedding:4b`: embedding/retrieval utility model role
- `qwen3.5:4b`: baseline generation + judge model
- `granite4.1:3b`: baseline generation + judge + fine-tuning base
- `glm-ocr`: OCR extension role

Fine-tuning base model (HF): `ibm-granite/granite-4.1-3b`

Alternative fallback base model (HF): `Qwen/Qwen3.5-4B`

## Dataset and Ground-Truth Constraints

Source: `coastalcph/lex_glue`

Used subsets:

- `ledgar`
- `unfair_tos`
- `scotus`
- `eurlex`
- `case_hold`

Artifact-backed dataset report: `data/processed/dataset_report.json`

Current processed row counts:

- train: `1540`
- validation: `97`
- test: `337`

Policy for targets:

- real LexGLUE data only
- deterministic mapping from labels/text cues to structured legal outputs
- no synthetic replacement dataset

## TRL, PEFT, Unsloth: What Was Actually Used

Implementation: `src/legal_clause_analyzer/finetune/train.py`

- TRL used (`SFTTrainer`, `SFTConfig`)
- PEFT used (`prepare_model_for_kbit_training`, `LoraConfig`, `get_peft_model`)
- Unsloth attempted first but not installed in this environment

Runtime proof: `artifacts/models/finetuned_adapter/training_backend_report.json`

- `backend_order`: `['unsloth', 'trl']`
- `selected_backend`: `trl_peft`
- `selected_model`: `ibm-granite/granite-4.1-3b`

## Current Measured Results (Artifact-Backed, 2026-06-22)

Sources:

- `artifacts/metrics/system_metrics.json`
- `artifacts/metrics/scoreboard.csv`
- `artifacts/metrics/judge_report.json`
- `artifacts/metrics/latency_report.json`
- `artifacts/metrics/hallucination_report.json`

Primary baseline comparator: `few_shot` (same top metrics as other non-fine-tuned baselines in this run).

| Metric | Baseline (`few_shot`) | Fine-Tuned Adapter | Delta |
|---|---:|---:|---:|
| ROUGE-L | 0.0023148148 | **0.0253431362** | **+0.0230283214** |
| BERTScore F1 | 0.8335364461 | **0.8351356983** | **+0.0015992522** |
| Risk F1 | 0.1085271318 | **0.1502192982** | **+0.0416921665** |
| Obligation F1 | **0.2500000000** | **0.2500000000** | 0.0000000000 |
| Liability F1 | 0.0000000000 | 0.0000000000 | 0.0000000000 |
| Red-Flag F1 | **1.0000000000** | **1.0000000000** | 0.0000000000 |

### LLM-as-a-Judge (Mean Overall)

- Granite judge overall:
  - baseline (`few_shot`): `2.7222222222`
  - fine-tuned: `2.7777777778`
  - delta: `+0.0555555556`
- Qwen judge overall:
  - baseline (`few_shot`): `3.0`
  - fine-tuned: `3.0`
  - delta: `0.0`

### Latency (Mean Seconds)

- `granite_zero_shot`: `1.5934249199`
- `few_shot`: `2.3021680878`
- `qwen_zero_shot`: `2.3150713524`
- `prompt_only`: `2.3928487653`
- `fine_tuned_adapter`: `7.9773901738`

### Hallucination Diagnostics

For all systems in this run (including fine-tuned):

- `unsupported_claim_rate = 0.0`
- `missing_obligation_rate = 1.0`
- `missing_liability_rate = 1.0`

Interpretation: unsupported claims are controlled in this setup, but obligation/liability recall remains a key failure area.

## Important Observed Behavior in Inference Examples

Source: `artifacts/inference/inference_examples.json`

- inference strategy: `fine_tuned_adapter` for all examples
- analysis payloads show parser fallback style outputs (`"No summary generated."`, empty arrays) in the shipped examples

Implication: strict path and adapter loading are working, but generation quality on these examples still needs improvement.

## Training Runtime and Resource Snapshot

Source: `artifacts/metrics/training_runtime_report.json`

- `cuda_available`: `true`
- elapsed seconds: `146.639812707901`
- max CUDA memory allocated: `5549631488` bytes
- max CUDA memory reserved: `5775556608` bytes

Training metrics source: `artifacts/models/finetuned_adapter/training_metrics.json`

## Project Layout

```text
Legal-Document-Summarizer-Clause-Analyzer/
├── configs/
├── data/
├── artifacts/
├── docs/
├── notebooks/
├── scripts/
├── src/legal_clause_analyzer/
└── tests/
```

## Run Locally

### Environment

```bash
cd /home/ahmad/AI/Legal-Document-Summarizer-Clause-Analyzer
uv venv --python 3.12.10
source .venv/bin/activate
uv sync
```

### Models

```bash
ollama pull qwen3-embedding:4b
ollama pull qwen3.5:4b
ollama pull granite4.1:3b
ollama pull glm-ocr
```

### Full strict run

```bash
source .venv/bin/activate
python scripts/run_all.py --config configs/default.yaml --strict-live
```

## Tutorial and Documentation Index

- Documentation index: `docs/README.md`
- Full handbook: `docs/HANDBOOK.md`
- Zero-to-hero chapter tutorials: `docs/tutorials/`
- PDF documentation: `docs/documentation.pdf`

## Reuse Targets

This component is designed to be reused in:

- legal research assistants
- contract review systems
- compliance monitoring systems
- legal risk scoring pipelines
- regulatory intelligence platforms

## Limitations and Next Steps

Current gaps shown by real outputs:

- liability extraction remains weak (`F1 = 0.0`)
- obligation recall remains weak (`missing_obligation_rate = 1.0`)
- inference example quality indicates parser fallback behavior still dominates

Priority next improvements:

1. stronger obligation/liability supervision strategy
2. richer long-context legal chunking + aggregation
3. schema-constrained decoding improvements
4. calibrated abstention for uncertain cases

## References Used for Implementation Decisions

- TRL docs: https://huggingface.co/docs/trl/sft_trainer
- PEFT quantization guide: https://huggingface.co/docs/peft/main/developer_guides/quantization
- Unsloth docs: https://docs.unsloth.ai/
- LexGLUE dataset card: https://huggingface.co/datasets/coastalcph/lex_glue

## Legal Disclaimer

This project is for technical demonstration and workflow support. It is not legal advice.
