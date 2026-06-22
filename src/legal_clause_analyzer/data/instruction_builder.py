"""Transform normalized LexGLUE rows into instruction/response supervision rows."""

from __future__ import annotations

from dataclasses import asdict

from legal_clause_analyzer.data.lexglue_loader import RawExample
from legal_clause_analyzer.data.policy_mapping import (
    infer_liabilities,
    infer_obligations,
    infer_red_flags,
    infer_restrictions,
    infer_rights,
    infer_risk_level,
)
from legal_clause_analyzer.schemas import EvidenceSpan, LegalAnalysis, SupervisedRow


def _dataset_domain(dataset_name: str) -> str:
    """Return a high-level legal hierarchy tag for explainability."""

    mapping = {
        "ledgar": "contract law",
        "unfair_tos": "consumer contract regulation",
        "scotus": "us case law",
        "eurlex": "eu regulatory law",
        "case_hold": "us legal reasoning",
    }
    return mapping.get(dataset_name, "general legal")


def _build_instruction(dataset_name: str) -> str:
    """Select user-facing instruction per sub-task."""

    if dataset_name in {"ledgar", "unfair_tos"}:
        return (
            "Analyze this legal clause. Provide a plain-English executive summary, key obligations, "
            "rights, restrictions, liabilities, risk level, red flags, and review recommendations."
        )

    if dataset_name in {"scotus", "eurlex", "case_hold"}:
        return (
            "Summarize this legal text for executives and legal operations teams. "
            "Then identify risks, liabilities, obligations, and potential red flags."
        )

    return "Summarize and analyze this legal text in structured form."


def _build_summary(row: RawExample) -> str:
    """Create deterministic summary target from gold labels/metadata."""

    if row.dataset == "case_hold":
        return row.metadata.get("correct_ending", "")[:420]

    labels = ", ".join(row.label_names[:3])
    domain = _dataset_domain(row.dataset)
    return (
        f"This {domain} text is primarily associated with: {labels}. "
        "The clause/document should be reviewed for obligations, enforceability, and risk allocation."
    )[:420]


def _build_bullets(row: RawExample, risk_level: str, liabilities: list[str]) -> list[str]:
    """Create concise executive bullets from deterministic features."""

    label_segment = ", ".join(row.label_names[:4])
    bullets = [
        f"Legal domain: {_dataset_domain(row.dataset)}.",
        f"Primary legal labels: {label_segment}.",
        f"Risk profile: {risk_level.upper()}.",
        f"Potential liability exposure: {', '.join(liabilities)}.",
    ]
    return bullets


def _build_clause_explanations(row: RawExample) -> list[str]:
    """Create short clause-level explanations aligned to gold labels."""

    output: list[str] = []
    for name in row.label_names[:4]:
        output.append(
            f"Label '{name}' indicates this text likely governs responsibilities, legal scope, "
            "or enforceability around that topic."
        )
    return output


def _evidence_spans(row: RawExample, obligations: list[str], restrictions: list[str]) -> list[EvidenceSpan]:
    """Collect evidence snippets to reduce hallucination risk."""

    spans: list[EvidenceSpan] = []
    for item in (obligations + restrictions)[:3]:
        spans.append(EvidenceSpan(text=item, rationale="Keyword-based legal cue from source text."))

    if not spans:
        spans.append(EvidenceSpan(text=row.text[:180], rationale="Opening legal context snippet."))
    return spans


def row_to_supervision(row: RawExample, policy: dict) -> SupervisedRow:
    """Build one structured supervision row from a raw legal sample."""

    risk_level = infer_risk_level(row.label_names, row.text, policy)
    liabilities = infer_liabilities(row.label_names, row.text, policy)
    obligations = infer_obligations(row.text, policy)
    rights = infer_rights(row.text, policy)
    restrictions = infer_restrictions(row.text, policy)
    red_flags = infer_red_flags(row.label_names, row.text, policy)

    analysis = LegalAnalysis(
        executive_summary=_build_summary(row),
        executive_bullets=_build_bullets(row, risk_level=risk_level, liabilities=liabilities),
        clause_explanations=_build_clause_explanations(row),
        risk_level=risk_level,
        risk_rationale=(
            "Risk level inferred from gold label semantics and explicit legal cue words in the clause text."
        ),
        liabilities=liabilities,
        obligations=obligations,
        rights=rights,
        restrictions=restrictions,
        red_flags=red_flags,
        recommended_review_areas=[
            "Confirm liability cap language.",
            "Validate termination and dispute-resolution clauses.",
            "Check compliance obligations against jurisdiction-specific rules.",
        ],
        evidence_spans=_evidence_spans(row, obligations=obligations, restrictions=restrictions),
    )

    return SupervisedRow(
        row_id=row.row_id,
        dataset=row.dataset,
        split=row.split,
        instruction=_build_instruction(row.dataset),
        input_text=row.text,
        output=analysis,
    )


def to_json_ready(row: SupervisedRow) -> dict:
    """Serialize `SupervisedRow` with nested models to a plain dictionary."""

    payload = asdict(row)
    payload["output"] = row.output.model_dump()
    return payload
