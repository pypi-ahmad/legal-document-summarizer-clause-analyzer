# Tutorial 02: Dataset and Supervision Design

## Learning Goals

- understand exactly how real LexGLUE data is ingested
- understand deterministic label-to-schema mapping
- connect data design decisions to observed model behavior

## What Is This Technique?

### Definition

Deterministic supervision design maps raw dataset labels and text cues to a unified instruction-response schema for training and evaluation.

### Why It Is Used

LexGLUE tasks are diverse and label spaces differ. A unified schema is required to train one model for executive summary + risk + extraction tasks.

### How It Appears in the Code

- dataset loader and split handling: `src/legal_clause_analyzer/data/lexglue_loader.py`
- policy-based mapping functions: `src/legal_clause_analyzer/data/policy_mapping.py`
- supervision row builder: `src/legal_clause_analyzer/data/instruction_builder.py`
- persisted dataset report + distributions: `scripts/prepare_data.py`

### Practical Explanation in This Project

`prepare_data.py` writes `train_supervised.jsonl`, `validation_supervised.jsonl`, and `test_supervised.jsonl` and a `dataset_report.json` that captures row counts, legal category hierarchy, and limitations.

## Dataset Ground Truth

Source dataset: `coastalcph/lex_glue`

Used subsets:

- `ledgar`
- `unfair_tos`
- `scotus`
- `eurlex`
- `case_hold`

Current processed row counts (`data/processed/dataset_report.json`):

- train: `1540`
- validation: `97`
- test: `337`

## Labeling Schema and Hierarchy

From artifact report:

- single-label datasets: `ledgar`, `scotus`, `case_hold`
- multi-label datasets: `unfair_tos`, `eurlex`

Regulatory hierarchy used in docs/report:

- contract law: `ledgar`, `unfair_tos`
- US case law: `scotus`, `case_hold`
- EU regulatory law: `eurlex`

## Long-Text Handling in Current Implementation

### What Is This Technique?

Conservative truncation and normalization for local runtime stability.

### Why It Is Used

Prevents memory and latency blow-ups on local hardware.

### How It Appears in the Code

- text normalization and truncation: `clean_text` in `lexglue_loader.py`
- max input char control: `configs/default.yaml` (`sampling.max_input_chars`)

### Practical Note

This is a pragmatic local strategy, not a perfect long-document strategy. For stronger long-context legal performance, add chunking/sliding windows + aggregation in a future version.

## Dataset Limitations (Artifact-Backed)

From `dataset_report.json`:

- no native gold executive-summary labels for all tasks
- no direct gold risk/liability taxonomy for unified schema
- cross-jurisdiction semantics can be underrepresented in compact model training

## Key Takeaway

The supervision layer is auditable and deterministic, which improves reproducibility, but it is also the main bottleneck behind extraction recall limits.
