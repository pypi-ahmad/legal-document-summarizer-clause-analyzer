"""Prepare deterministic instruction-response data from real LexGLUE splits."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from legal_clause_analyzer.data.instruction_builder import row_to_supervision
from legal_clause_analyzer.data.lexglue_loader import build_dataset_registry
from legal_clause_analyzer.data.policy_mapping import load_policy
from legal_clause_analyzer.settings import load_config
from legal_clause_analyzer.utils.io import write_json, write_jsonl
from legal_clause_analyzer.utils.logging_utils import configure_logging
from legal_clause_analyzer.utils.seed import seed_everything


def _distribution_plot(rows: list[dict], output_path: Path) -> None:
    """Save dataset distribution chart for tutorial and reporting."""

    df = pd.DataFrame(rows)
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    df["dataset"].value_counts().plot(kind="bar", ax=axes[0], title="Rows per Dataset")
    df["risk_level"].value_counts().plot(kind="bar", ax=axes[1], title="Risk Level Distribution")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build supervised legal analysis dataset from LexGLUE")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--policy", default="configs/policy_mapping.yaml")
    args = parser.parse_args()

    configure_logging()
    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)
    policy = load_policy(args.policy)

    registry = build_dataset_registry(cfg)

    split_rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for dataset_name, split_map in registry.items():
        for split_name, examples in split_map.items():
            for raw in examples:
                row = row_to_supervision(raw, policy=policy)
                payload = row.model_dump()
                payload["risk_level"] = row.output.risk_level
                split_rows[split_name].append(payload)

    processed_dir = Path(cfg.paths.processed_dir).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    for split_name, rows in split_rows.items():
        write_jsonl(processed_dir / f"{split_name}_supervised.jsonl", rows)
        logger.info("Wrote {} rows to {}", len(rows), f"{split_name}_supervised.jsonl")

    # Persist dataset report for README/notebooks.
    report = {
        "row_counts": {split: len(rows) for split, rows in split_rows.items()},
        "datasets_per_split": {
            split: dict(Counter(row["dataset"] for row in rows)) for split, rows in split_rows.items()
        },
        "risk_distribution_train": dict(
            Counter(row["output"]["risk_level"] for row in split_rows["train"])
        ),
        "dataset_structure": {
            "ledgar": "single-label contract clause classification",
            "unfair_tos": "multi-label unfair consumer contract terms",
            "scotus": "single-label US Supreme Court opinion topics",
            "eurlex": "multi-label EU legal act classification",
            "case_hold": "single-label legal holding selection",
        },
        "labeling_schema": {
            "single_label_datasets": ["ledgar", "scotus", "case_hold"],
            "multi_label_datasets": ["unfair_tos", "eurlex"],
        },
        "regulatory_hierarchy": {
            "contract_law": ["ledgar", "unfair_tos"],
            "case_law_us": ["scotus", "case_hold"],
            "regulatory_law_eu": ["eurlex"],
        },
        "case_metadata": {
            "case_hold": ["correct_ending", "endings"],
            "others": ["dataset_name", "split", "label_names"],
        },
        "dataset_limitations": [
            "No native executive-summary gold labels across all tasks.",
            "No direct gold risk/liability taxonomy for our unified output schema.",
            "Cross-jurisdiction semantics can be underrepresented in compact model training.",
        ],
    }
    write_json(processed_dir / "dataset_report.json", report)

    flat_rows = []
    for split_name, rows in split_rows.items():
        for row in rows:
            flat_rows.append(
                {
                    "split": split_name,
                    "dataset": row["dataset"],
                    "risk_level": row["output"]["risk_level"],
                }
            )
    _distribution_plot(flat_rows, Path(cfg.paths.figures_dir).resolve() / "dataset_distributions.png")


if __name__ == "__main__":
    main()
