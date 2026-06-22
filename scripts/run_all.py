"""Single entry point for end-to-end project execution."""

from __future__ import annotations

import argparse
import subprocess
import sys

from loguru import logger


def _run(cmd: list[str]) -> None:
    if cmd and cmd[0] == "python":
        cmd = [sys.executable, *cmd[1:]]
    logger.info("Running: {}", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full legal analyzer pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-notebooks", action="store_true")
    parser.add_argument("--skip-finetuned-eval", action="store_true")
    parser.add_argument(
        "--strict-live",
        action="store_true",
        help="Fail on any fallback/reuse path and require live model outputs.",
    )
    args = parser.parse_args()

    cfg = args.config
    _run(["python", "scripts/prepare_data.py", "--config", cfg])
    baselines_cmd = ["python", "scripts/run_baselines.py", "--config", cfg]
    train_cmd = ["python", "scripts/train_model.py", "--config", cfg]
    if args.strict_live:
        baselines_cmd.append("--strict-live")
        train_cmd.append("--strict-live")
    _run(baselines_cmd)
    _run(train_cmd)

    eval_cmd = ["python", "scripts/evaluate.py", "--config", cfg]
    if args.skip_finetuned_eval:
        eval_cmd.append("--skip-finetuned")
    if args.strict_live:
        eval_cmd.append("--strict-live")
    _run(eval_cmd)

    inference_cmd = ["python", "scripts/run_inference.py", "--config", cfg]
    if args.strict_live:
        inference_cmd.append("--strict-live")
    _run(inference_cmd)
    if not args.skip_notebooks:
        _run(["python", "scripts/generate_notebooks.py"])
        _run(["python", "scripts/execute_notebooks.py", "--notebooks-dir", "notebooks"])

    if args.strict_live:
        validate_cmd = ["python", "scripts/validate_strict_run.py", "--config", cfg]
        if not args.skip_notebooks:
            validate_cmd.append("--require-notebooks")
        _run(validate_cmd)


if __name__ == "__main__":
    main()
