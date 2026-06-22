"""Evaluate baseline and fine-tuned systems, including LLM-as-a-judge."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from legal_clause_analyzer.baselines.model_clients import GenerationConfig
from legal_clause_analyzer.evaluation.hallucination import hallucination_metrics
from legal_clause_analyzer.evaluation.judge import JudgeSpec, run_judge
from legal_clause_analyzer.evaluation.metrics import evaluate_predictions, flatten_metrics_for_table
from legal_clause_analyzer.evaluation.plots import plot_metric_bars, plot_risk_confusion, plot_system_scoreboard
from legal_clause_analyzer.finetune.infer_adapter import generate_with_adapter, load_finetuned_model
from legal_clause_analyzer.settings import load_config
from legal_clause_analyzer.utils.io import read_jsonl, write_json, write_jsonl
from legal_clause_analyzer.utils.logging_utils import configure_logging


def _collect_baseline_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("predictions_*.jsonl"))


def _evaluate_file(path: Path) -> tuple[str, list[dict], dict]:
    rows = read_jsonl(path)
    name = path.stem.replace("predictions_", "")
    metrics = evaluate_predictions(rows)
    return name, rows, metrics


def _run_judges(cfg, rows: list[dict], strict_live: bool = False) -> dict:
    judges = [
        JudgeSpec(
            name="granite_judge",
            backend=cfg.runtime.backend,
            model=cfg.models.generation_granite_ollama,
        ),
        JudgeSpec(
            name="qwen_judge",
            backend=cfg.runtime.backend,
            model=cfg.models.generation_qwen_ollama,
        ),
    ]
    subset = rows[: cfg.evaluation.judge_max_examples]

    gen_cfg = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        max_tokens=260,
        num_ctx=cfg.runtime.num_ctx,
    )

    all_scores: dict[str, list[dict]] = {judge.name: [] for judge in judges}
    for row in tqdm(subset, desc="Judge evaluation"):
        for judge in judges:
            try:
                score = run_judge(
                    spec=judge,
                    input_text=row["input_text"],
                    reference_json=row["reference"],
                    prediction_json=row["prediction"],
                    generation_cfg=gen_cfg,
                )
                all_scores[judge.name].append(score)
            except Exception as exc:
                if strict_live:
                    raise RuntimeError(f"Judge {judge.name} failed on {row['row_id']}: {exc}") from exc
                logger.warning("Judge {} failed on {}: {}", judge.name, row["row_id"], exc)

    summary: dict[str, dict[str, float]] = {}
    for judge_name, scores in all_scores.items():
        if strict_live and len(scores) != len(subset):
            raise RuntimeError(
                f"Judge {judge_name} produced {len(scores)} scores for {len(subset)} rows in strict live mode."
            )
        if not scores:
            summary[judge_name] = {}
            continue
        table = pd.DataFrame(scores)
        summary[judge_name] = {col: float(table[col].mean()) for col in table.columns}
    return summary


def _run_finetuned_predictions(cfg, test_rows: list[dict], run_dir: Path) -> Path:
    adapter_dir = Path(cfg.training.output_dir).resolve()
    model = None
    tokenizer = None
    last_exc: Exception | None = None
    for candidate in [cfg.models.finetune_base_hf, cfg.models.alt_finetune_base_hf]:
        try:
            model, tokenizer = load_finetuned_model(
                candidate,
                adapter_dir=adapter_dir,
                max_memory_gpu_gib=cfg.training.max_memory_gpu_gib,
                max_memory_cpu_gib=cfg.training.max_memory_cpu_gib,
                bf16=cfg.training.bf16,
                use_4bit=cfg.training.use_4bit,
            )
            logger.info("Loaded fine-tuned adapter with base model: {}", candidate)
            break
        except Exception as exc:
            last_exc = exc
            logger.warning("Failed loading fine-tuned inference model {}: {}", candidate, exc)

    if model is None or tokenizer is None:
        raise RuntimeError(f"No usable fine-tuned inference model found: {last_exc}")

    outputs: list[dict] = []
    for row in tqdm(test_rows[: cfg.evaluation.max_eval_examples], desc="Fine-tuned inference"):
        started = time.perf_counter()
        pred = generate_with_adapter(
            model=model,
            tokenizer=tokenizer,
            text=row["input_text"],
            max_new_tokens=cfg.runtime.max_generation_tokens,
            temperature=cfg.runtime.temperature,
            top_p=cfg.runtime.top_p,
        )
        latency_seconds = time.perf_counter() - started
        outputs.append(
            {
                "row_id": row["row_id"],
                "dataset": row["dataset"],
                "input_text": row["input_text"],
                "reference": row["output"],
                "prediction": pred.model_dump(),
                "system": "fine_tuned_adapter",
                "latency_seconds": latency_seconds,
            }
        )

    out_path = run_dir / "predictions_fine_tuned_adapter.jsonl"
    write_jsonl(out_path, outputs)
    return out_path


def _latency_summary(rows: list[dict]) -> dict[str, float]:
    latencies = [float(row.get("latency_seconds", 0.0)) for row in rows]
    if not latencies:
        return {"mean_seconds": 0.0, "median_seconds": 0.0, "p95_seconds": 0.0}
    series = pd.Series(latencies, dtype=float)
    return {
        "mean_seconds": float(series.mean()),
        "median_seconds": float(series.median()),
        "p95_seconds": float(series.quantile(0.95)),
    }


def _build_scoreboard(metrics_df: pd.DataFrame) -> pd.DataFrame:
    selected: list[dict] = []
    mapping = [
        ("summarization", "rougeL", "summ_rougeL"),
        ("summarization", "bertscore_f1", "summ_bertscore_f1"),
        ("risk_classification", "f1", "risk_f1"),
        ("obligation_extraction", "f1", "obligation_f1"),
        ("liability_extraction", "f1", "liability_f1"),
        ("red_flag_extraction", "f1", "red_flag_f1"),
    ]
    for system in sorted(metrics_df["system"].unique()):
        system_df = metrics_df[metrics_df["system"] == system]
        for family, metric_column, metric_name in mapping:
            family_df = system_df[system_df["metric_family"] == family]
            if family_df.empty or metric_column not in family_df.columns:
                continue
            score = float(family_df.iloc[0][metric_column])
            selected.append({"system": system, "metric": metric_name, "score": score})
    return pd.DataFrame(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate legal analyzer systems")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-finetuned", action="store_true")
    parser.add_argument(
        "--strict-live",
        action="store_true",
        help="Require live fine-tuned + judge outputs; fail if any stage is skipped/empty.",
    )
    args = parser.parse_args()

    configure_logging()
    cfg = load_config(args.config)

    run_dir = Path(cfg.paths.run_dir).resolve() / cfg.project.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    test_rows = read_jsonl(Path(cfg.paths.processed_dir).resolve() / "test_supervised.jsonl")
    if not args.skip_finetuned:
        try:
            _run_finetuned_predictions(cfg, test_rows=test_rows, run_dir=run_dir)
        except Exception as exc:
            if args.strict_live:
                raise RuntimeError(f"Strict live mode requires fine-tuned predictions: {exc}") from exc
            logger.warning("Skipping fine-tuned predictions due to runtime failure: {}", exc)

    metric_tables: list[pd.DataFrame] = []
    all_rows_by_system: dict[str, list[dict]] = {}
    metric_report: dict[str, dict] = {}

    for file in _collect_baseline_files(run_dir):
        system_name, rows, metrics = _evaluate_file(file)
        all_rows_by_system[system_name] = rows
        metric_report[system_name] = metrics

        table = flatten_metrics_for_table(metrics)
        table.insert(0, "system", system_name)
        metric_tables.append(table)

    if not metric_tables:
        raise RuntimeError("No prediction files found. Run baselines first.")

    metrics_df = pd.concat(metric_tables, ignore_index=True)
    metrics_csv = Path(cfg.paths.metrics_dir).resolve() / "system_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # Aggregate hallucination + judge reports per system.
    hallucination_report = {
        system: hallucination_metrics(rows[: cfg.evaluation.hallucination_max_examples])
        for system, rows in all_rows_by_system.items()
    }
    judge_report = {system: _run_judges(cfg, rows, strict_live=args.strict_live) for system, rows in all_rows_by_system.items()}

    if args.strict_live:
        for system, judges in judge_report.items():
            for judge_name, judge_metrics in judges.items():
                if not judge_metrics:
                    raise RuntimeError(
                        f"Strict live mode requires non-empty judge metrics: system={system}, judge={judge_name}"
                    )

    write_json(Path(cfg.paths.metrics_dir).resolve() / "system_metrics.json", metric_report)
    write_json(Path(cfg.paths.metrics_dir).resolve() / "hallucination_report.json", hallucination_report)
    write_json(Path(cfg.paths.metrics_dir).resolve() / "judge_report.json", judge_report)
    latency_report = {system: _latency_summary(rows) for system, rows in all_rows_by_system.items()}
    write_json(Path(cfg.paths.metrics_dir).resolve() / "latency_report.json", latency_report)

    # Visualizations.
    main_system = "fine_tuned_adapter" if "fine_tuned_adapter" in set(metrics_df["system"]) else metrics_df[
        "system"
    ].iloc[0]
    plot_metric_bars(
        metrics_df[metrics_df["system"] == main_system].drop(columns=["system"]),
        Path(cfg.paths.figures_dir).resolve() / "metric_bars.png",
    )

    scoreboard = _build_scoreboard(metrics_df)
    if not scoreboard.empty:
        scoreboard_csv = Path(cfg.paths.metrics_dir).resolve() / "scoreboard.csv"
        scoreboard.to_csv(scoreboard_csv, index=False)
        plot_system_scoreboard(scoreboard, Path(cfg.paths.figures_dir).resolve() / "system_scoreboard.png")

    # Risk confusion for fine-tuned system if available, else qwen_zero_shot.
    confusion_system = "fine_tuned_adapter"
    if confusion_system not in all_rows_by_system:
        confusion_system = "qwen_zero_shot"

    rows = all_rows_by_system[confusion_system]
    y_true = [row["reference"]["risk_level"] for row in rows]
    y_pred = [row["prediction"]["risk_level"] for row in rows]
    plot_risk_confusion(y_true, y_pred, Path(cfg.paths.figures_dir).resolve() / "risk_confusion_matrix.png")


if __name__ == "__main__":
    main()
