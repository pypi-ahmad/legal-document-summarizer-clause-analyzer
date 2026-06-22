"""Visualization helpers for training and evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


sns.set_theme(style="whitegrid")


def plot_training_curve(curve_csv: Path, out_path: Path) -> None:
    """Plot training/eval loss over steps from trainer logs."""

    df = pd.read_csv(curve_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    if "loss" in df.columns and "step" in df.columns:
        train_df = df.dropna(subset=["loss", "step"])
        ax.plot(train_df["step"], train_df["loss"], label="train_loss", linewidth=2)
    if "eval_loss" in df.columns and "step" in df.columns:
        eval_df = df.dropna(subset=["eval_loss", "step"])
        ax.plot(eval_df["step"], eval_df["eval_loss"], label="eval_loss", linewidth=2)

    ax.set_title("Fine-Tuning Loss Curves")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_risk_confusion(y_true: list[str], y_pred: list[str], out_path: Path) -> None:
    """Plot risk-level confusion matrix."""

    labels = ["low", "medium", "high"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Risk Classification Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_metric_bars(metric_table: pd.DataFrame, out_path: Path) -> None:
    """Plot grouped bars for metric families."""

    melted = metric_table.melt(id_vars=["metric_family"], var_name="metric", value_name="score")
    melted = melted[melted["metric"] != "accuracy"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="metric_family", y="score", hue="metric", ax=ax)
    ax.set_title("Evaluation Metrics by Family")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_system_scoreboard(scoreboard: pd.DataFrame, out_path: Path) -> None:
    """Plot compact per-system score comparison from pre-selected metrics."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=scoreboard, x="metric", y="score", hue="system", ax=ax)
    ax.set_title("System Comparison on Key Metrics")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
