"""Run inference examples and persist structured outputs."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from legal_clause_analyzer.inference.pipeline import analyze_legal_text
from legal_clause_analyzer.settings import load_config
from legal_clause_analyzer.utils.io import write_json
from legal_clause_analyzer.utils.logging_utils import configure_logging


EXAMPLES = [
    "The Vendor shall indemnify, defend, and hold harmless the Client from all third-party claims arising out of Vendor services. The Client may terminate this Agreement upon material breach with 15 days notice.",
    "The service provider may unilaterally change fees and terms at any time without prior notice. Liability shall be excluded to the fullest extent permitted by law.",
    "Any dispute arising under this agreement must be resolved through arbitration in New York. Each party is responsible for compliance with all applicable data protection regulations.",
]


def _release_ollama_gpu_memory() -> None:
    """Unload currently resident Ollama models to reduce VRAM pressure."""

    try:
        output = subprocess.check_output(["ollama", "ps"], text=True).strip().splitlines()
    except Exception:
        return

    if len(output) <= 1:
        return

    for line in output[1:]:
        model = line.split()[0].strip() if line.strip() else ""
        if model and model != "NAME":
            subprocess.run(["ollama", "stop", model], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run legal analysis inference examples")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--strict-live",
        action="store_true",
        help="Require fine-tuned live inference and fail if fallback paths are hit.",
    )
    args = parser.parse_args()

    configure_logging()
    cfg = load_config(args.config)
    _release_ollama_gpu_memory()

    outputs = []
    for idx, text in enumerate(EXAMPLES, start=1):
        started = time.perf_counter()
        result = analyze_legal_text(text=text, cfg=cfg, use_few_shot=True, strict_live=args.strict_live)
        if args.strict_live and result.strategy != "fine_tuned_adapter":
            raise RuntimeError(
                f"Strict live mode expected fine_tuned_adapter strategy, got {result.strategy} for example {idx}"
            )
        latency_seconds = time.perf_counter() - started
        outputs.append(
            {
                "example_id": idx,
                "input_text": text,
                "model_used": result.model_used,
                "strategy": result.strategy,
                "latency_seconds": latency_seconds,
                "analysis": result.analysis.model_dump(),
            }
        )

    out_path = Path(cfg.paths.inference_dir).resolve() / "inference_examples.json"
    write_json(out_path, outputs)


if __name__ == "__main__":
    main()
