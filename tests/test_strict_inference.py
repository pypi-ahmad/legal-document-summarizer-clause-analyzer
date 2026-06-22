from __future__ import annotations

from pathlib import Path

import pytest

from legal_clause_analyzer.inference import pipeline
from legal_clause_analyzer.settings import load_config


def _base_cfg() -> object:
    cfg = load_config("configs/default.yaml")
    return cfg


def test_strict_live_requires_adapter_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg.training.output_dir = str(tmp_path / "missing_adapter_dir")

    with pytest.raises(RuntimeError, match="Strict live mode requires fine-tuned adapter inference"):
        pipeline.analyze_legal_text("legal text", cfg=cfg, use_few_shot=True, strict_live=True)


def test_non_strict_inference_uses_static_fallback_on_backend_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _base_cfg()
    cfg.training.output_dir = str(tmp_path / "missing_adapter_dir")

    def _raise_backend_error(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(pipeline, "run_baseline_generation", _raise_backend_error)

    result = pipeline.analyze_legal_text("legal text", cfg=cfg, use_few_shot=True, strict_live=False)
    assert result.strategy == "fallback_static"
    assert result.model_used == "deterministic_fallback"
