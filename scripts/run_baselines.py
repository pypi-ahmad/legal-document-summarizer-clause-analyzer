"""Run baseline systems on held-out legal evaluation rows."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from legal_clause_analyzer.baselines.generation import BaselineSpec, run_baseline_generation
from legal_clause_analyzer.baselines.model_clients import GenerationConfig
from legal_clause_analyzer.schemas import LegalAnalysis
from legal_clause_analyzer.settings import load_config
from legal_clause_analyzer.utils.io import read_jsonl, write_jsonl
from legal_clause_analyzer.utils.logging_utils import configure_logging


def _baseline_specs(cfg) -> list[BaselineSpec]:
    return [
        BaselineSpec(
            name="prompt_only",
            backend=cfg.runtime.backend,
            model=cfg.models.generation_qwen_ollama,
            few_shot=False,
            style="prompt_only",
        ),
        BaselineSpec(
            name="few_shot",
            backend=cfg.runtime.backend,
            model=cfg.models.generation_qwen_ollama,
            few_shot=True,
            style="prompt_only",
        ),
        BaselineSpec(
            name="granite_zero_shot",
            backend=cfg.runtime.backend,
            model=cfg.models.generation_granite_ollama,
            few_shot=False,
            style="granite_zero_shot",
        ),
        BaselineSpec(
            name="qwen_zero_shot",
            backend=cfg.runtime.backend,
            model=cfg.models.generation_qwen_ollama,
            few_shot=False,
            style="qwen_zero_shot",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline model systems")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--strict-live",
        action="store_true",
        help="Fail on any model error; do not emit deterministic fallback outputs.",
    )
    args = parser.parse_args()

    configure_logging()
    cfg = load_config(args.config)

    test_rows = read_jsonl(Path(cfg.paths.processed_dir).resolve() / "test_supervised.jsonl")
    test_rows = test_rows[: cfg.evaluation.max_eval_examples]

    gen_cfg = GenerationConfig(
        temperature=cfg.runtime.temperature,
        top_p=cfg.runtime.top_p,
        max_tokens=cfg.runtime.max_generation_tokens,
        num_ctx=cfg.runtime.num_ctx,
    )

    run_dir = Path(cfg.paths.run_dir).resolve() / cfg.project.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    def _fallback_analysis(text: str) -> LegalAnalysis:
        """Deterministic fallback when model inference is temporarily unavailable."""

        snippet = text[:220]
        return LegalAnalysis(
            executive_summary=f"Fallback summary from source snippet: {snippet}",
            executive_bullets=["Model call failed; generated deterministic fallback output."],
            clause_explanations=["Fallback path does not include model-driven legal interpretation."],
            risk_level="medium",
            risk_rationale="Fallback default due inference failure.",
            liabilities=["contractual liability"],
            obligations=[],
            rights=[],
            restrictions=[],
            red_flags=[],
            recommended_review_areas=["Re-run model inference for full analysis."],
            evidence_spans=[{"text": snippet, "rationale": "source snippet fallback"}],
        )

    for spec in _baseline_specs(cfg):
        outputs: list[dict] = []
        for row in tqdm(test_rows, desc=f"Baseline {spec.name}"):
            started = time.perf_counter()
            try:
                pred = run_baseline_generation(spec=spec, input_text=row["input_text"], generation_cfg=gen_cfg)
            except Exception as exc:
                if args.strict_live:
                    raise RuntimeError(f"Baseline {spec.name} failed on {row['row_id']}: {exc}") from exc
                logger.warning("Baseline {} failed on {}: {}", spec.name, row["row_id"], exc)
                # Keep the pipeline executable with deterministic fallback output.
                pred = _fallback_analysis(row["input_text"])
            latency_seconds = time.perf_counter() - started

            outputs.append(
                {
                    "row_id": row["row_id"],
                    "dataset": row["dataset"],
                    "input_text": row["input_text"],
                    "reference": row["output"],
                    "prediction": pred.model_dump(),
                    "system": spec.name,
                    "latency_seconds": latency_seconds,
                }
            )

        out_path = run_dir / f"predictions_{spec.name}.jsonl"
        write_jsonl(out_path, outputs)
        logger.info("Saved baseline predictions: {}", out_path)


if __name__ == "__main__":
    main()
