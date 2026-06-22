"""Validate that a run satisfies strict live end-to-end criteria."""

from __future__ import annotations

import argparse
from pathlib import Path

from legal_clause_analyzer.settings import load_config
from legal_clause_analyzer.utils.io import read_jsonl
from legal_clause_analyzer.utils.logging_utils import configure_logging


FALLBACK_SUMMARY_PREFIX = "Fallback summary from source snippet:"
EXPECTED_SYSTEMS = [
    "prompt_only",
    "few_shot",
    "granite_zero_shot",
    "qwen_zero_shot",
    "fine_tuned_adapter",
]


def _fail_if(cond: bool, message: str, errors: list[str]) -> None:
    if cond:
        errors.append(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate strict live run artifacts")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--require-notebooks", action="store_true")
    args = parser.parse_args()

    configure_logging()
    cfg = load_config(args.config)

    errors: list[str] = []
    root = Path(".").resolve()
    run_dir = Path(cfg.paths.run_dir).resolve() / cfg.project.run_name

    # 1) Prediction files must exist for all systems and contain zero fallback rows.
    for system in EXPECTED_SYSTEMS:
        pred_path = run_dir / f"predictions_{system}.jsonl"
        _fail_if(not pred_path.exists(), f"Missing predictions file: {pred_path}", errors)
        if not pred_path.exists():
            continue
        rows = read_jsonl(pred_path)
        _fail_if(len(rows) == 0, f"Empty predictions file: {pred_path}", errors)
        fallback_rows = [
            row["row_id"]
            for row in rows
            if str(row.get("prediction", {}).get("executive_summary", "")).startswith(FALLBACK_SUMMARY_PREFIX)
        ]
        _fail_if(
            len(fallback_rows) > 0,
            f"Fallback predictions detected in {pred_path.name}: {len(fallback_rows)} rows",
            errors,
        )

    # 2) Judge report must be non-empty for every system and judge.
    judge_path = Path(cfg.paths.metrics_dir).resolve() / "judge_report.json"
    _fail_if(not judge_path.exists(), f"Missing judge report: {judge_path}", errors)
    if judge_path.exists():
        import json

        judge_report = json.loads(judge_path.read_text())
        for system in EXPECTED_SYSTEMS:
            system_scores = judge_report.get(system, {})
            _fail_if(not system_scores, f"Missing judge system entry: {system}", errors)
            for judge_name in ["granite_judge", "qwen_judge"]:
                metrics = system_scores.get(judge_name, {})
                _fail_if(
                    not metrics,
                    f"Empty judge metrics for system={system}, judge={judge_name}",
                    errors,
                )

    # 3) Training runtime must indicate CUDA availability.
    runtime_path = Path(cfg.paths.metrics_dir).resolve() / "training_runtime_report.json"
    _fail_if(not runtime_path.exists(), f"Missing training runtime report: {runtime_path}", errors)
    if runtime_path.exists():
        import json

        runtime_report = json.loads(runtime_path.read_text())
        cuda_available = bool(runtime_report.get("torch_memory", {}).get("cuda_available", False))
        _fail_if(not cuda_available, "training_runtime_report indicates cuda_available=false", errors)

    # 4) Inference must be fine-tuned strategy only.
    inference_path = Path(cfg.paths.inference_dir).resolve() / "inference_examples.json"
    _fail_if(not inference_path.exists(), f"Missing inference examples: {inference_path}", errors)
    if inference_path.exists():
        import json

        inference_rows = json.loads(inference_path.read_text())
        _fail_if(len(inference_rows) == 0, "Inference examples file is empty", errors)
        bad_rows = [
            row
            for row in inference_rows
            if row.get("strategy") != "fine_tuned_adapter" or row.get("model_used") == "deterministic_fallback"
        ]
        _fail_if(
            len(bad_rows) > 0,
            f"Inference strict violations detected: {len(bad_rows)} rows are not fine_tuned_adapter",
            errors,
        )

    # 5) Optional notebook execution check.
    if args.require_notebooks:
        notebook_dir = root / "notebooks"
        source_notebooks = sorted(
            path
            for path in notebook_dir.glob("*.ipynb")
            if not path.name.endswith(".executed.ipynb")
        )
        expected_outputs = [path.with_name(path.stem + ".executed.ipynb") for path in source_notebooks]
        missing = [path for path in expected_outputs if not path.exists()]
        _fail_if(
            len(missing) > 0,
            f"Missing executed notebooks: {', '.join(str(path.name) for path in missing)}",
            errors,
        )

    if errors:
        message = "\n".join(f"- {item}" for item in errors)
        raise RuntimeError(f"Strict run validation failed:\n{message}")

    print("strict_run_validation_passed")


if __name__ == "__main__":
    main()
