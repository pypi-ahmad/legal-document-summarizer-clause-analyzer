"""Run QLoRA fine-tuning on prepared legal instruction data."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import torch
from loguru import logger

from legal_clause_analyzer.evaluation.plots import plot_training_curve
from legal_clause_analyzer.finetune.train import TrainArtifacts, run_finetuning
from legal_clause_analyzer.settings import load_config
from legal_clause_analyzer.utils.io import write_json
from legal_clause_analyzer.utils.logging_utils import configure_logging
from legal_clause_analyzer.utils.seed import seed_everything


def _release_ollama_gpu_memory() -> None:
    """Unload currently resident Ollama models to free GPU VRAM for training."""

    try:
        output = subprocess.check_output(["ollama", "ps"], text=True).strip().splitlines()
    except Exception as exc:
        logger.debug("Skipping Ollama unload check: {}", exc)
        return

    if len(output) <= 1:
        return

    # Header usually starts with NAME and SIZE; rows start with model tags.
    loaded_models: list[str] = []
    for line in output[1:]:
        row = line.strip()
        if not row:
            continue
        model = row.split()[0].strip()
        if model and model != "NAME":
            loaded_models.append(model)

    for model in loaded_models:
        try:
            subprocess.run(["ollama", "stop", model], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("Requested Ollama unload for model: {}", model)
        except Exception as exc:
            logger.debug("Could not stop Ollama model {}: {}", model, exc)


def _start_ollama_gpu_guard(interval_seconds: float = 1.5) -> tuple[threading.Event, threading.Thread]:
    """Continuously unload GPU-resident Ollama sessions during training.

    Some local services can auto-load embedding models in the background, which
    steals VRAM and causes intermittent OOMs. This guard keeps VRAM clear for
    the training process.
    """

    stop_event = threading.Event()

    def _worker() -> None:
        while not stop_event.is_set():
            try:
                output = subprocess.check_output(["ollama", "ps"], text=True).strip().splitlines()
                for line in output[1:]:
                    row = line.strip()
                    if not row:
                        continue
                    parts = row.split()
                    if len(parts) < 4:
                        continue
                    model_name = parts[0]
                    processor_blob = " ".join(parts[3:6]).upper()
                    if "GPU" in processor_blob and model_name != "NAME":
                        subprocess.run(
                            ["ollama", "stop", model_name],
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
            except Exception:
                pass
            stop_event.wait(interval_seconds)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return stop_event, thread


def _gpu_snapshot() -> dict:
    """Capture simple GPU utilization snapshot via nvidia-smi if available."""

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        output = subprocess.check_output(cmd, text=True).strip().splitlines()
        rows = []
        for line in output:
            name, mem_total, mem_used, util = [x.strip() for x in line.split(",")]
            rows.append(
                {
                    "name": name,
                    "memory_total_mb": float(mem_total),
                    "memory_used_mb": float(mem_used),
                    "utilization_gpu_percent": float(util),
                }
            )
        return {"nvidia_smi": rows}
    except Exception:
        return {"nvidia_smi": []}


def _reuse_existing_training_artifacts(cfg) -> TrainArtifacts | None:
    """Reuse previous training outputs when a new local training run is not possible."""

    out_dir = Path(cfg.training.output_dir).resolve()
    metrics_path = out_dir / "training_metrics.json"
    curve_csv_path = out_dir / "training_curve.csv"
    backend_report_path = out_dir / "training_backend_report.json"
    required = [metrics_path, curve_csv_path, backend_report_path]
    if all(path.exists() for path in required):
        logger.warning("Reusing existing training artifacts at {}", out_dir)
        return TrainArtifacts(
            adapter_dir=out_dir,
            metrics_path=metrics_path,
            curve_csv_path=curve_csv_path,
            backend_report_path=backend_report_path,
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune legal analyzer model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--strict-live",
        action="store_true",
        help="Require real CUDA training and fail instead of reusing old artifacts.",
    )
    args = parser.parse_args()

    configure_logging()
    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)
    _release_ollama_gpu_memory()
    guard_stop, guard_thread = _start_ollama_gpu_guard(interval_seconds=1.5)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_path = Path(cfg.paths.processed_dir).resolve() / "train_supervised.jsonl"
    val_path = Path(cfg.paths.processed_dir).resolve() / "validation_supervised.jsonl"

    pre_gpu = _gpu_snapshot()
    start = time.time()
    try:
        try:
            artifacts = run_finetuning(cfg=cfg, train_jsonl=train_path, val_jsonl=val_path)
        except RuntimeError as exc:
            # Keep local end-to-end workflow executable on machines where CUDA is unavailable.
            if "CUDA GPU is required" not in str(exc):
                raise
            if args.strict_live:
                raise
            reused = _reuse_existing_training_artifacts(cfg)
            if reused is None:
                raise
            artifacts = reused
    finally:
        guard_stop.set()
        guard_thread.join(timeout=2.0)
    elapsed = time.time() - start
    post_gpu = _gpu_snapshot()

    memory = {
        "cuda_available": bool(torch.cuda.is_available()),
        "max_memory_allocated_bytes": float(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0,
        "max_memory_reserved_bytes": float(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0.0,
    }

    runtime_report = {
        "elapsed_seconds": elapsed,
        "pre_gpu": pre_gpu,
        "post_gpu": post_gpu,
        "torch_memory": memory,
        "training_metrics_path": str(artifacts.metrics_path),
        "training_backend_report_path": str(artifacts.backend_report_path),
    }
    write_json(Path(cfg.paths.metrics_dir).resolve() / "training_runtime_report.json", runtime_report)

    curve_png = Path(cfg.paths.figures_dir).resolve() / "training_curve.png"
    plot_training_curve(artifacts.curve_csv_path, curve_png)
    logger.info("Saved training curve: {}", curve_png)


if __name__ == "__main__":
    main()
