"""Fine-tuning pipeline using TRL + PEFT with optional Unsloth acceleration.

References:
- TRL SFTTrainer docs: https://huggingface.co/docs/trl/sft_trainer
- PEFT quantization + QLoRA guide: https://huggingface.co/docs/peft/main/developer_guides/quantization
- Unsloth docs: https://docs.unsloth.ai/
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer

from legal_clause_analyzer.finetune.dataset_format import format_for_sft
from legal_clause_analyzer.schemas import SupervisedRow
from legal_clause_analyzer.settings import AppConfig
from legal_clause_analyzer.utils.io import read_jsonl, write_json


@dataclass
class TrainArtifacts:
    """Returned artifact locations after training."""

    adapter_dir: Path
    metrics_path: Path
    curve_csv_path: Path
    backend_report_path: Path


@dataclass
class ModelBundle:
    """In-memory model pack ready for TRL training."""

    model: Any
    tokenizer: Any
    backend: str
    model_name: str
    quantized_4bit: bool


def _load_supervised_rows(path: Path) -> list[SupervisedRow]:
    rows = read_jsonl(path)
    return [SupervisedRow.model_validate(row) for row in rows]


def _dataset_from_rows(rows: list[SupervisedRow]) -> Dataset:
    return Dataset.from_dict({"text": [format_for_sft(row) for row in rows]})


def _torch_dtype(cfg: AppConfig) -> torch.dtype:
    return torch.bfloat16 if cfg.training.bf16 else torch.float16


def _resolve_max_memory(cfg: AppConfig) -> dict[int | str, str]:
    return {
        0: f"{cfg.training.max_memory_gpu_gib:.1f}GiB",
        "cpu": f"{cfg.training.max_memory_cpu_gib:.1f}GiB",
    }


def _build_lora_config(cfg: AppConfig) -> LoraConfig:
    return LoraConfig(
        r=cfg.training.lora_r,
        lora_alpha=cfg.training.lora_alpha,
        lora_dropout=cfg.training.lora_dropout,
        target_modules=cfg.training.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def _build_sft_config(cfg: AppConfig, output_dir: Path) -> SFTConfig:
    gradient_checkpointing_kwargs: dict[str, Any] | None = None
    if cfg.training.gradient_checkpointing:
        gradient_checkpointing_kwargs = {
            "use_reentrant": cfg.training.gradient_checkpointing_reentrant,
        }

    return SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        num_train_epochs=cfg.training.num_train_epochs,
        max_steps=cfg.training.max_steps,
        logging_steps=cfg.training.logging_steps,
        eval_steps=cfg.training.eval_steps,
        save_steps=cfg.training.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=cfg.training.bf16,
        fp16=not cfg.training.bf16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        report_to=[],
        seed=cfg.project.seed,
        dataloader_pin_memory=True,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        max_length=cfg.training.max_seq_length,
        dataset_text_field="text",
        packing=False,
        remove_unused_columns=True,
    )


def _candidate_models(cfg: AppConfig) -> list[str]:
    # First choice: Granite 3B for lower memory footprint on an 8GB-class GPU.
    # Fallback: Qwen 4B for stronger general generation quality if Granite fails.
    return [cfg.models.finetune_base_hf, cfg.models.alt_finetune_base_hf]


def _backend_order(cfg: AppConfig) -> list[str]:
    if cfg.training.trainer_backend == "unsloth":
        return ["unsloth"]
    if cfg.training.trainer_backend == "trl":
        return ["trl"]
    return ["unsloth", "trl"]


def _load_trl_peft_bundle(cfg: AppConfig, model_name: str, offload_dir: Path) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if cfg.training.use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_torch_dtype(cfg),
            llm_int8_enable_fp32_cpu_offload=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        max_memory=_resolve_max_memory(cfg),
        offload_folder=str(offload_dir),
        quantization_config=quant_cfg,
        torch_dtype=_torch_dtype(cfg),
    )
    model.config.use_cache = False

    # PEFT docs recommend preparing k-bit models before LoRA attachment.
    if cfg.training.use_4bit:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, _build_lora_config(cfg))
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        backend="trl_peft",
        model_name=model_name,
        quantized_4bit=cfg.training.use_4bit,
    )


def _load_unsloth_bundle(cfg: AppConfig, model_name: str) -> ModelBundle:
    try:
        from unsloth import FastLanguageModel  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Unsloth is not installed in this environment.") from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=cfg.training.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.training.use_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.training.lora_r,
        lora_alpha=cfg.training.lora_alpha,
        lora_dropout=cfg.training.lora_dropout,
        target_modules=cfg.training.lora_target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth" if cfg.training.gradient_checkpointing else False,
    )
    model.config.use_cache = False
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        backend="unsloth_trl",
        model_name=model_name,
        quantized_4bit=cfg.training.use_4bit,
    )


def _build_trainer(
    cfg: AppConfig,
    bundle: ModelBundle,
    output_dir: Path,
    train_ds: Dataset,
    val_ds: Dataset,
) -> SFTTrainer:
    data_collator = DataCollatorForLanguageModeling(tokenizer=bundle.tokenizer, mlm=False)
    return SFTTrainer(
        model=bundle.model,
        args=_build_sft_config(cfg, output_dir=output_dir),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=bundle.tokenizer,
        data_collator=data_collator,
    )


def run_finetuning(cfg: AppConfig, train_jsonl: Path, val_jsonl: Path) -> TrainArtifacts:
    """Execute instruction fine-tuning and persist adapter + logs.

    The runtime selects backends in order:
    1. `unsloth` (if requested/available)
    2. `trl` + `peft` fallback
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this local fine-tuning profile.")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    train_rows = _load_supervised_rows(train_jsonl)
    val_rows = _load_supervised_rows(val_jsonl)
    train_ds = _dataset_from_rows(train_rows)
    val_ds = _dataset_from_rows(val_rows)

    out_dir = Path(cfg.training.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    offload_dir = out_dir / "offload_cache"
    offload_dir.mkdir(parents=True, exist_ok=True)

    attempts: list[dict[str, str]] = []
    require_unsloth = cfg.training.require_unsloth
    backend_order = _backend_order(cfg)

    for backend in backend_order:
        for model_name in _candidate_models(cfg):
            try:
                logger.info("Loading backend={} model={}", backend, model_name)
                if backend == "unsloth":
                    bundle = _load_unsloth_bundle(cfg, model_name=model_name)
                else:
                    bundle = _load_trl_peft_bundle(cfg, model_name=model_name, offload_dir=offload_dir)

                trainer = _build_trainer(
                    cfg=cfg,
                    bundle=bundle,
                    output_dir=out_dir,
                    train_ds=train_ds,
                    val_ds=val_ds,
                )

                logger.info(
                    "Starting fine-tuning with backend={} model={} train_rows={} val_rows={}",
                    bundle.backend,
                    bundle.model_name,
                    len(train_rows),
                    len(val_rows),
                )
                train_result = trainer.train()
                eval_result = trainer.evaluate()

                trainer.save_model(str(out_dir))
                bundle.tokenizer.save_pretrained(str(out_dir))

                metrics = {
                    "train": {k: float(v) for k, v in train_result.metrics.items()},
                    "eval": {k: float(v) for k, v in eval_result.items()},
                    "model_name": bundle.model_name,
                    "training_backend": bundle.backend,
                    "quantized_4bit": bundle.quantized_4bit,
                    "train_rows": len(train_rows),
                    "validation_rows": len(val_rows),
                }
                metrics_path = out_dir / "training_metrics.json"
                write_json(metrics_path, metrics)

                history = pd.DataFrame(trainer.state.log_history)
                curve_csv_path = out_dir / "training_curve.csv"
                history.to_csv(curve_csv_path, index=False)

                backend_report = {
                    "backend_order": backend_order,
                    "selected_backend": bundle.backend,
                    "selected_model": bundle.model_name,
                    "attempts": attempts,
                }
                backend_report_path = out_dir / "training_backend_report.json"
                write_json(backend_report_path, backend_report)

                return TrainArtifacts(
                    adapter_dir=out_dir,
                    metrics_path=metrics_path,
                    curve_csv_path=curve_csv_path,
                    backend_report_path=backend_report_path,
                )
            except Exception as exc:  # pragma: no cover - runtime fallback path
                msg = str(exc)
                attempts.append({"backend": backend, "model": model_name, "error": msg[:500]})
                logger.warning("Failed backend={} model={} -> {}", backend, model_name, msg)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if backend == "unsloth" and require_unsloth:
            raise RuntimeError("Unsloth was required but all Unsloth loading attempts failed.")

    raise RuntimeError(f"All training attempts failed. Attempts: {attempts}")
