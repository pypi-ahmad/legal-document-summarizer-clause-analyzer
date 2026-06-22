"""Evaluation metrics for summarization, classification, and extraction."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
from bert_score import score as bertscore_score
from loguru import logger
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sacrebleu import sentence_bleu
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from legal_clause_analyzer.schemas import LegalAnalysis


def _safe_average(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarization_metrics(references: list[str], predictions: list[str]) -> dict[str, float]:
    """Compute ROUGE/BLEU/METEOR/BERTScore for summary text."""

    if not references:
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "bleu": 0.0,
            "meteor": 0.0,
            "bertscore_f1": 0.0,
        }

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1: list[float] = []
    rouge2: list[float] = []
    rougeL: list[float] = []
    bleu: list[float] = []
    meteor: list[float] = []

    for ref, pred in zip(references, predictions, strict=False):
        scores = scorer.score(ref, pred)
        rouge1.append(scores["rouge1"].fmeasure)
        rouge2.append(scores["rouge2"].fmeasure)
        rougeL.append(scores["rougeL"].fmeasure)
        bleu.append(sentence_bleu(pred, [ref]).score / 100.0)
        meteor.append(meteor_score([ref.split()], pred.split()))

    # BERTScore is expensive; run on same list but catch transient issues.
    bert_f1 = 0.0
    try:
        _, _, f1 = bertscore_score(predictions, references, lang="en", verbose=False)
        bert_f1 = float(f1.mean().item())
    except Exception as exc:
        logger.warning("BERTScore failed, continuing with 0.0: {}", exc)

    return {
        "rouge1": _safe_average(rouge1),
        "rouge2": _safe_average(rouge2),
        "rougeL": _safe_average(rougeL),
        "bleu": _safe_average(bleu),
        "meteor": _safe_average(meteor),
        "bertscore_f1": bert_f1,
    }


def classification_metrics(y_true: list[str], y_pred: list[str], average: str = "macro") -> dict[str, float]:
    """Compute Accuracy/Precision/Recall/F1 for categorical labels."""

    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )
    acc = accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _set_f1(ref_items: list[str], pred_items: list[str]) -> tuple[float, float, float]:
    """Set-level precision/recall/F1 for extraction outputs."""

    ref = {x.strip().lower() for x in ref_items if x.strip()}
    pred = {x.strip().lower() for x in pred_items if x.strip()}

    if not ref and not pred:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0

    tp = len(ref & pred)
    precision = tp / max(1, len(pred))
    recall = tp / max(1, len(ref))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def extraction_metrics(reference: list[list[str]], prediction: list[list[str]]) -> dict[str, float]:
    """Compute micro-style extraction metrics across examples."""

    ps: list[float] = []
    rs: list[float] = []
    f1s: list[float] = []

    for ref_items, pred_items in zip(reference, prediction, strict=False):
        p, r, f1 = _set_f1(ref_items, pred_items)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)

    return {
        "precision": _safe_average(ps),
        "recall": _safe_average(rs),
        "f1": _safe_average(f1s),
    }


def evaluate_predictions(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Aggregate all metric families from prediction rows.

    Each row must include:
    - `reference`: `LegalAnalysis`
    - `prediction`: `LegalAnalysis`
    """

    references = [LegalAnalysis.model_validate(row["reference"]) for row in rows]
    predictions = [LegalAnalysis.model_validate(row["prediction"]) for row in rows]

    summary_refs = [r.executive_summary for r in references]
    summary_preds = [p.executive_summary for p in predictions]

    risk_refs = [r.risk_level for r in references]
    risk_preds = [p.risk_level for p in predictions]

    obligations_ref = [r.obligations for r in references]
    obligations_pred = [p.obligations for p in predictions]

    liabilities_ref = [r.liabilities for r in references]
    liabilities_pred = [p.liabilities for p in predictions]

    red_flags_ref = [r.red_flags for r in references]
    red_flags_pred = [p.red_flags for p in predictions]

    return {
        "summarization": summarization_metrics(summary_refs, summary_preds),
        "risk_classification": classification_metrics(risk_refs, risk_preds),
        "obligation_extraction": extraction_metrics(obligations_ref, obligations_pred),
        "liability_extraction": extraction_metrics(liabilities_ref, liabilities_pred),
        "red_flag_extraction": extraction_metrics(red_flags_ref, red_flags_pred),
    }


def flatten_metrics_for_table(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert nested metric dictionary into tabular format."""

    rows: list[dict] = []
    for family, values in metrics.items():
        row = {"metric_family": family}
        row.update(values)
        rows.append(row)
    return pd.DataFrame(rows)
