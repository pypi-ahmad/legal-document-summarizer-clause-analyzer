"""Load and normalize LexGLUE datasets used in the project.

The loader preserves original split boundaries to avoid data leakage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from loguru import logger

from legal_clause_analyzer.settings import AppConfig


_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class RawExample:
    """Normalized sample that hides dataset-specific quirks."""

    row_id: str
    dataset: str
    split: str
    text: str
    labels: list[int]
    label_names: list[str]
    metadata: dict[str, Any]


def clean_text(text: str, max_chars: int) -> str:
    """Apply conservative normalization for legal text."""

    text = _WHITESPACE_RE.sub(" ", text.strip())
    return text[:max_chars]


def _resolve_label_names(ds: Dataset, dataset_name: str, labels: list[int]) -> list[str]:
    """Map integer labels to readable names from Hugging Face feature metadata."""

    if dataset_name in {"scotus", "ledgar"}:
        names = ds.features["label"].names
        return [str(names[label]) for label in labels]

    if dataset_name in {"eurlex", "unfair_tos", "ecthr_a", "ecthr_b"}:
        names = ds.features["labels"].feature.names
        return [str(names[label]) for label in labels]

    # case_hold uses multiple-choice labels (0..4); we keep index names.
    if dataset_name == "case_hold":
        names = ds.features["label"].names
        return [str(names[label]) for label in labels]

    return [f"label_{label}" for label in labels]


def _extract_text(dataset_name: str, row: dict[str, Any]) -> str:
    """Select the text field per dataset."""

    if dataset_name in {"scotus", "eurlex", "ledgar", "unfair_tos"}:
        return str(row["text"])

    if dataset_name in {"ecthr_a", "ecthr_b"}:
        return "\n".join([str(part) for part in row["text"]])

    if dataset_name == "case_hold":
        return str(row["context"])

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _extract_labels(dataset_name: str, row: dict[str, Any]) -> list[int]:
    """Normalize labels into integer list representation."""

    if dataset_name in {"scotus", "ledgar", "case_hold"}:
        return [int(row["label"])]

    if dataset_name in {"eurlex", "unfair_tos", "ecthr_a", "ecthr_b"}:
        return [int(x) for x in row["labels"]]

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _sample_dataset(ds: Dataset, max_rows: int, seed: int) -> Dataset:
    """Deterministically sample rows for local runtime constraints."""

    if len(ds) <= max_rows:
        return ds
    return ds.shuffle(seed=seed).select(range(max_rows))


def load_lexglue_split(
    cfg: AppConfig,
    dataset_name: str,
    split: str,
    max_rows: int,
) -> list[RawExample]:
    """Load and normalize one dataset split.

    Args:
        cfg: Typed app config.
        dataset_name: LexGLUE config name, e.g. `ledgar`.
        split: `train`, `validation`, or `test`.
        max_rows: Upper bound for local runtime.

    Returns:
        List of `RawExample` rows.
    """

    cache_dir = Path(cfg.project.cache_dir).resolve()
    ds = load_dataset(cfg.project.hf_repo, dataset_name, split=split, cache_dir=str(cache_dir))
    ds = _sample_dataset(ds, max_rows=max_rows, seed=cfg.project.seed)

    rows: list[RawExample] = []
    for idx, row in enumerate(ds):
        labels = _extract_labels(dataset_name, row)
        label_names = _resolve_label_names(ds, dataset_name, labels)
        text = clean_text(_extract_text(dataset_name, row), max_chars=cfg.sampling.max_input_chars)
        if len(text) < 80:
            continue

        metadata: dict[str, Any] = {}
        if dataset_name == "case_hold":
            metadata["endings"] = [str(x) for x in row["endings"]]
            metadata["correct_ending"] = str(row["endings"][int(row["label"])])

        rows.append(
            RawExample(
                row_id=f"{dataset_name}_{split}_{idx:06d}",
                dataset=dataset_name,
                split=split,
                text=text,
                labels=labels,
                label_names=label_names,
                metadata=metadata,
            )
        )

    logger.info("Loaded {} {} rows from split={} (requested <= {})", len(rows), dataset_name, split, max_rows)
    return rows


def build_dataset_registry(cfg: AppConfig) -> dict[str, dict[str, list[RawExample]]]:
    """Load all datasets/splits needed by this project.

    Returns:
        Nested structure: `{dataset_name: {split: [RawExample, ...]}}`
    """

    # This subset mix aligns with target capabilities:
    # - ledgar/unfair_tos: contract clause analysis + risk
    # - scotus/eurlex: long-form legal prose + domain transfer
    # - case_hold: concise holding references for summary-style supervision
    datasets = ["ledgar", "unfair_tos", "scotus", "eurlex", "case_hold"]

    split_limits = {
        "train": cfg.sampling.train_rows_per_dataset,
        "validation": cfg.sampling.val_rows_per_dataset,
        "test": cfg.sampling.test_rows_per_dataset,
    }

    registry: dict[str, dict[str, list[RawExample]]] = {}
    for dataset_name in datasets:
        registry[dataset_name] = {}
        for split, limit in split_limits.items():
            registry[dataset_name][split] = load_lexglue_split(
                cfg=cfg,
                dataset_name=dataset_name,
                split=split,
                max_rows=limit,
            )

    return registry
