# Tutorial 04: Fine-Tuning with TRL, PEFT, and Unsloth

## Learning Goals

- understand LoRA/QLoRA in this codebase
- understand exactly why TRL and PEFT are required
- understand where Unsloth is optional and what happened in this run

## What Is This Technique?

### Definition

QLoRA fine-tuning trains small adapter parameters on top of a quantized base model, rather than full-model parameter updates.

### Why It Is Used

It makes local fine-tuning feasible on limited VRAM GPUs.

### How It Appears in the Code

- training pipeline core: `src/legal_clause_analyzer/finetune/train.py`
- trainer entrypoint and GPU guard logic: `scripts/train_model.py`
- training format: `src/legal_clause_analyzer/finetune/dataset_format.py`

### Practical Explanation in This Project

The training stack tries Unsloth first (if available), then falls back to TRL+PEFT. In this environment, Unsloth was unavailable, and training completed with `trl_peft`.

## TRL Coverage

### Definition

TRL provides `SFTTrainer` and `SFTConfig` for supervised instruction tuning.

### Why used here

Needed for stable trainer lifecycle: logging, eval, save, and metric traces.

### Where used

`SFTTrainer`/`SFTConfig` usage in `train.py`.

### Post-run impact

Produced structured training artifacts:

- `training_metrics.json`
- `training_curve.csv`
- adapter checkpoint and metadata files

## PEFT Coverage

### Definition

PEFT (Parameter-Efficient Fine-Tuning) attaches LoRA adapters.

### Why used here

Avoids full-parameter fine-tuning cost on local hardware.

### Where used

- `prepare_model_for_kbit_training`
- `LoraConfig`
- `get_peft_model`

all in `train.py`.

### Post-run impact

Fine-tuned adapter artifacts created under `artifacts/models/finetuned_adapter/` with `quantized_4bit=true` in training metrics.

## Unsloth Coverage

### Definition

Optional accelerated backend for LoRA/QLoRA flows.

### Why considered

Potential speed and memory improvements.

### Where used

Optional path `_load_unsloth_bundle` in `train.py`.

### Post-run impact

Source: `artifacts/models/finetuned_adapter/training_backend_report.json`

- Unsloth attempts recorded
- both attempts failed with "Unsloth is not installed in this environment"
- selected backend: `trl_peft`

## Real Training Runtime Snapshot

Source: `artifacts/metrics/training_runtime_report.json`

- CUDA available: `true`
- elapsed: `146.639812707901s`
- max allocated: `5549631488` bytes
- max reserved: `5775556608` bytes

## Key Takeaway

TRL+PEFT is the required backbone in this environment; Unsloth is an optimization path, not a hard dependency.
