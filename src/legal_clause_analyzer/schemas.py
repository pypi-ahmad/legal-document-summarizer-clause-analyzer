"""Structured response schemas for model outputs and references."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvidenceSpan(BaseModel):
    """Evidence snippet with a light explanation."""

    text: str
    rationale: str


class LegalAnalysis(BaseModel):
    """Canonical output schema used in training/evaluation/inference."""

    executive_summary: str
    executive_bullets: list[str] = Field(default_factory=list)
    clause_explanations: list[str] = Field(default_factory=list)
    risk_level: str
    risk_rationale: str
    liabilities: list[str] = Field(default_factory=list)
    obligations: list[str] = Field(default_factory=list)
    rights: list[str] = Field(default_factory=list)
    restrictions: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    recommended_review_areas: list[str] = Field(default_factory=list)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)


class SupervisedRow(BaseModel):
    """Instruction-tuning sample row."""

    row_id: str
    dataset: str
    split: str
    instruction: str
    input_text: str
    output: LegalAnalysis
