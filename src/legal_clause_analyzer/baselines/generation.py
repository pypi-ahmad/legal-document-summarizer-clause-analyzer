"""Baseline generation systems and prompt templates."""

from __future__ import annotations

from dataclasses import dataclass

from legal_clause_analyzer.baselines.model_clients import GenerationConfig, UnifiedModelClient, try_parse_json
from legal_clause_analyzer.schemas import LegalAnalysis


SYSTEM_PROMPT = (
    "You are a legal analyst assistant. Return a single valid JSON object only with keys: "
    "executive_summary, executive_bullets, clause_explanations, risk_level, risk_rationale, "
    "liabilities, obligations, rights, restrictions, red_flags, recommended_review_areas, evidence_spans. "
    "Never include markdown, comments, or prose outside the JSON object."
)


@dataclass
class BaselineSpec:
    """Baseline recipe and model binding."""

    name: str
    backend: str
    model: str
    few_shot: bool
    style: str


def _few_shot_block() -> str:
    """Compact few-shot demonstration for structured legal outputs."""

    return (
        "Example 1\n"
        "Input: The supplier shall indemnify and hold harmless the client for third-party claims.\n"
        "Output: {\"executive_summary\":\"Supplier assumes third-party claim exposure.\","
        "\"executive_bullets\":[\"Indemnification duty exists.\",\"Financial exposure may be material.\"],"
        "\"clause_explanations\":[\"The clause shifts defense obligations to the supplier.\"],"
        "\"risk_level\":\"medium\",\"risk_rationale\":\"Broad indemnity language without cap.\","
        "\"liabilities\":[\"financial liability\"],\"obligations\":[\"Supplier must indemnify the client.\"],"
        "\"rights\":[],\"restrictions\":[],\"red_flags\":[\"broad indemnification\"],"
        "\"recommended_review_areas\":[\"Add liability cap and defense scope limits.\"],"
        "\"evidence_spans\":[{\"text\":\"shall indemnify and hold harmless\",\"rationale\":\"obligation phrase\"}]}\n"
    )


def build_prompt(input_text: str, style: str, few_shot: bool) -> str:
    """Construct prompt variant by baseline style."""

    style_line = {
        "prompt_only": "Use concise legal plain English.",
        "granite_zero_shot": "Use conservative legal reasoning and avoid unsupported claims.",
        "qwen_zero_shot": "Use structured JSON and include operational recommendations.",
    }.get(style, "Use structured JSON output.")

    head = (
        "Task: Analyze the legal text and return structured JSON.\n"
        f"Guideline: {style_line}\n"
        "Risk level must be one of: low, medium, high.\n"
        "If uncertain, choose medium and explain why in risk_rationale.\n"
        "Output JSON schema:\n"
        "{\n"
        '  "executive_summary": "string",\n'
        '  "executive_bullets": ["string"],\n'
        '  "clause_explanations": ["string"],\n'
        '  "risk_level": "low|medium|high",\n'
        '  "risk_rationale": "string",\n'
        '  "liabilities": ["string"],\n'
        '  "obligations": ["string"],\n'
        '  "rights": ["string"],\n'
        '  "restrictions": ["string"],\n'
        '  "red_flags": ["string"],\n'
        '  "recommended_review_areas": ["string"],\n'
        '  "evidence_spans": [{"text":"string","rationale":"string"}]\n'
        "}\n"
    )

    # Truncate to keep local inference latency practical on consumer hardware.
    clipped_input = input_text[:1600]
    body = (
        f"Input Legal Text:\n{clipped_input}\n\n"
        "Return strictly one JSON object only. Do not wrap with markdown code fences."
    )
    if few_shot:
        return f"{head}\n{_few_shot_block()}\n{body}"
    return f"{head}\n{body}"


def safe_analysis_from_text(raw: str) -> LegalAnalysis:
    """Parse model JSON response with robust fallback defaults."""

    parsed = try_parse_json(raw) or {}

    def _list(name: str) -> list[str]:
        value = parsed.get(name, [])
        if isinstance(value, list):
            return [str(x) for x in value]
        return []

    evidence_raw = parsed.get("evidence_spans", [])
    evidence: list[dict[str, str]] = []
    if isinstance(evidence_raw, list):
        for item in evidence_raw[:5]:
            if isinstance(item, dict):
                evidence.append(
                    {
                        "text": str(item.get("text", ""))[:220],
                        "rationale": str(item.get("rationale", ""))[:220],
                    }
                )

    risk_level = str(parsed.get("risk_level", "medium")).strip().lower()
    if risk_level not in {"low", "medium", "high"}:
        risk_level = "medium"

    return LegalAnalysis(
        executive_summary=str(parsed.get("executive_summary", "No summary generated."))[:500],
        executive_bullets=_list("executive_bullets"),
        clause_explanations=_list("clause_explanations"),
        risk_level=risk_level,
        risk_rationale=str(parsed.get("risk_rationale", "Model output parsing fallback."))[:400],
        liabilities=_list("liabilities"),
        obligations=_list("obligations"),
        rights=_list("rights"),
        restrictions=_list("restrictions"),
        red_flags=_list("red_flags"),
        recommended_review_areas=_list("recommended_review_areas"),
        evidence_spans=evidence,
    )


def run_baseline_generation(
    spec: BaselineSpec,
    input_text: str,
    generation_cfg: GenerationConfig,
) -> LegalAnalysis:
    """Generate structured legal analysis for one text sample."""

    client = UnifiedModelClient(backend=spec.backend, model=spec.model)
    prompt = build_prompt(input_text=input_text, style=spec.style, few_shot=spec.few_shot)
    raw = client.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT, config=generation_cfg)
    return safe_analysis_from_text(raw)
