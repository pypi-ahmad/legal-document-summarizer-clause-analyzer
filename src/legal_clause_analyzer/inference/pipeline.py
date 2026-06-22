"""Inference pipeline that returns a strict structured legal analysis JSON."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from legal_clause_analyzer.baselines.generation import BaselineSpec, run_baseline_generation
from legal_clause_analyzer.baselines.model_clients import GenerationConfig
from legal_clause_analyzer.finetune.infer_adapter import generate_with_adapter, load_finetuned_model
from legal_clause_analyzer.schemas import LegalAnalysis
from legal_clause_analyzer.settings import AppConfig

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class InferenceResult:
    """Top-level inference payload."""

    analysis: LegalAnalysis
    model_used: str
    strategy: str


def _fallback_analysis(text: str) -> LegalAnalysis:
    """Return deterministic output when all model backends are unavailable."""

    snippet = text[:220]
    return LegalAnalysis(
        executive_summary=f"Fallback summary from source snippet: {snippet}",
        executive_bullets=["Model call failed; generated deterministic fallback output."],
        clause_explanations=["Fallback path does not include model-driven legal interpretation."],
        risk_level="medium",
        risk_rationale="Fallback default due inference backend unavailability.",
        liabilities=["contractual liability"],
        obligations=[],
        rights=[],
        restrictions=[],
        red_flags=[],
        recommended_review_areas=["Re-run model inference for full analysis."],
        evidence_spans=[{"text": snippet, "rationale": "source snippet fallback"}],
    )


def analyze_legal_text(
    text: str,
    cfg: AppConfig,
    use_few_shot: bool = False,
    strict_live: bool = False,
) -> InferenceResult:
    """Run legal analysis using fine-tuned adapter first, then baseline fallback."""

    adapter_dir = Path(cfg.training.output_dir).resolve()
    if adapter_dir.exists():
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
                analysis = generate_with_adapter(
                    model=model,
                    tokenizer=tokenizer,
                    text=text,
                    max_new_tokens=cfg.runtime.max_generation_tokens,
                    temperature=cfg.runtime.temperature,
                    top_p=cfg.runtime.top_p,
                )
                return InferenceResult(
                    analysis=analysis,
                    model_used=candidate,
                    strategy="fine_tuned_adapter",
                )
            except Exception as exc:  # pragma: no cover - runtime fallback path
                logger.warning("Fine-tuned inference load failed for {}: {}", candidate, exc)
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if strict_live:
        raise RuntimeError("Strict live mode requires fine-tuned adapter inference; adapter load failed.")

    spec = BaselineSpec(
        name="inference_qwen",
        backend=cfg.runtime.backend,
        model=cfg.models.generation_qwen_ollama,
        few_shot=use_few_shot,
        style="qwen_zero_shot",
    )
    generation_cfg = GenerationConfig(
        temperature=cfg.runtime.temperature,
        top_p=cfg.runtime.top_p,
        max_tokens=cfg.runtime.max_generation_tokens,
        num_ctx=cfg.runtime.num_ctx,
    )
    try:
        analysis = run_baseline_generation(spec=spec, input_text=text, generation_cfg=generation_cfg)
        return InferenceResult(analysis=analysis, model_used=spec.model, strategy=spec.style)
    except Exception as exc:  # pragma: no cover - runtime safety fallback
        if strict_live:
            raise RuntimeError(f"Strict live inference fallback model call failed: {exc}") from exc
        logger.warning("Inference fallback model call failed: {}", exc)
        return InferenceResult(
            analysis=_fallback_analysis(text),
            model_used="deterministic_fallback",
            strategy="fallback_static",
        )
