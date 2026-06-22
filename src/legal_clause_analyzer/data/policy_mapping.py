"""Deterministic policy mapping from gold labels/text to legal analysis fields."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_policy(path: str | Path) -> dict:
    """Load YAML mapping rules used for auditable structured targets."""

    return yaml.safe_load(Path(path).read_text())


def _norm(text: str) -> str:
    return text.strip().lower()


def infer_risk_level(label_names: list[str], text: str, policy: dict) -> str:
    """Map labels/text to low/medium/high risk with deterministic precedence."""

    tokens = [_norm(x) for x in label_names]
    joined = _norm(text)

    for key in policy["risk_levels"].get("high", []):
        if key in joined or key in tokens:
            return "high"

    for key in policy["risk_levels"].get("medium", []):
        if key in joined or key in tokens:
            return "medium"

    return "low"


def infer_liabilities(label_names: list[str], text: str, policy: dict) -> list[str]:
    """Map labels/text to predefined liability buckets."""

    joined = _norm(text)
    label_tokens = {_norm(x) for x in label_names}

    buckets: list[str] = []
    for bucket, keys in policy["liability_buckets"].items():
        if any((k in joined) or (k in label_tokens) for k in keys):
            buckets.append(bucket)
    return buckets or ["contractual liability"]


def _extract_sentences_with_keywords(text: str, keywords: list[str], limit: int = 4) -> list[str]:
    """Collect short evidence snippets containing target keywords."""

    # Simple sentence split is sufficient for deterministic rule extraction here.
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    hits: list[str] = []
    for sentence in sentences:
        s_lower = sentence.lower()
        if any(key in s_lower for key in keywords):
            hits.append(sentence[:220])
        if len(hits) >= limit:
            break
    return hits


def infer_obligations(text: str, policy: dict) -> list[str]:
    """Extract obligation snippets from explicit modal language."""

    return _extract_sentences_with_keywords(text, policy["obligation_keywords"], limit=4)


def infer_rights(text: str, policy: dict) -> list[str]:
    """Extract rights snippets from permissive phrasing."""

    return _extract_sentences_with_keywords(text, policy["rights_keywords"], limit=4)


def infer_restrictions(text: str, policy: dict) -> list[str]:
    """Extract restrictions snippets from prohibitive language."""

    return _extract_sentences_with_keywords(text, policy["restriction_keywords"], limit=4)


def infer_red_flags(label_names: list[str], text: str, policy: dict) -> list[str]:
    """Flag explicit high-risk lexical patterns."""

    joined = _norm(text)
    labels = " ".join([_norm(x) for x in label_names])
    flags = [flag for flag in policy["red_flag_keywords"] if flag in joined or flag in labels]

    if "unilateral" in joined and not any("unilateral" in f for f in flags):
        flags.append("unilateral control clause")

    return flags[:5]
