"""Typed configuration for the legal summarizer project.

The project intentionally keeps configuration in YAML to make experiments readable
for beginners while still preserving production-grade typing and validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    """Top-level project metadata and data source settings."""

    seed: int = 42
    hf_repo: str = "coastalcph/lex_glue"
    cache_dir: str = "data/hf_cache"
    run_name: str = "balanced_local_v1"


class ModelConfig(BaseModel):
    """Model IDs used across embedding, generation, and fine-tuning."""

    embedding_ollama: str
    generation_qwen_ollama: str
    generation_granite_ollama: str
    ocr_ollama: str
    finetune_base_hf: str
    alt_finetune_base_hf: str


class RuntimeConfig(BaseModel):
    """Inference/runtime controls.

    `backend`:
    - `ollama`: use local Ollama server for inference/judging baselines.
    - `transformers`: use Hugging Face models directly.
    """

    backend: Literal["ollama", "transformers"] = "ollama"
    use_gpu: bool = True
    max_generation_tokens: int = 420
    temperature: float = 0.1
    top_p: float = 0.9
    num_ctx: int = 8192


class SamplingConfig(BaseModel):
    """Dataset sampling controls for balanced local runs."""

    train_rows_per_dataset: int = 2200
    val_rows_per_dataset: int = 400
    test_rows_per_dataset: int = 500
    max_input_chars: int = 10000
    max_output_chars: int = 2400


class TrainingConfig(BaseModel):
    """QLoRA/LoRA training knobs."""

    output_dir: str = "artifacts/models/finetuned_adapter"
    trainer_backend: Literal["auto", "trl", "unsloth"] = "auto"
    require_unsloth: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.08
    weight_decay: float = 0.01
    num_train_epochs: float = 1.0
    max_steps: int = 120
    eval_steps: int = 20
    save_steps: int = 40
    logging_steps: int = 5
    max_seq_length: int = 1536
    dataloader_num_workers: int = 2
    use_4bit: bool = True
    bf16: bool = True
    gradient_checkpointing: bool = True
    gradient_checkpointing_reentrant: bool = False
    max_memory_gpu_gib: float = 7.0
    max_memory_cpu_gib: float = 24.0


class EvaluationConfig(BaseModel):
    """Evaluation loop controls."""

    max_eval_examples: int = 160
    judge_max_examples: int = 60
    hallucination_max_examples: int = 120
    summary_dataset_priority: list[str] = Field(default_factory=lambda: ["case_hold", "scotus", "eurlex"])


class PathConfig(BaseModel):
    """Filesystem outputs used by scripts and notebooks."""

    processed_dir: str = "data/processed"
    run_dir: str = "artifacts/runs"
    figures_dir: str = "artifacts/figures"
    metrics_dir: str = "artifacts/metrics"
    inference_dir: str = "artifacts/inference"


class AppConfig(BaseModel):
    """Fully validated application config."""

    project: ProjectConfig
    models: ModelConfig
    runtime: RuntimeConfig
    sampling: SamplingConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    paths: PathConfig


def _ensure_dirs(root: Path, cfg: AppConfig) -> None:
    """Create configured directories eagerly.

    Args:
        root: Project root path.
        cfg: Parsed app config.
    """

    dirs = [
        cfg.project.cache_dir,
        cfg.paths.processed_dir,
        cfg.paths.run_dir,
        cfg.paths.figures_dir,
        cfg.paths.metrics_dir,
        cfg.paths.inference_dir,
        cfg.training.output_dir,
    ]
    for rel in dirs:
        (root / rel).mkdir(parents=True, exist_ok=True)


def load_config(config_path: str | Path) -> AppConfig:
    """Load YAML config and return a fully typed object.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Validated `AppConfig` instance.
    """

    path = Path(config_path).resolve()
    payload = yaml.safe_load(path.read_text())
    cfg = AppConfig.model_validate(payload)
    _ensure_dirs(path.parent.parent, cfg)
    return cfg
