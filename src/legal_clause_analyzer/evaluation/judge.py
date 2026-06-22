"""LLM-as-a-judge scoring for legal output quality."""

from __future__ import annotations

from dataclasses import dataclass

from legal_clause_analyzer.baselines.model_clients import GenerationConfig, UnifiedModelClient, try_parse_json


@dataclass
class JudgeSpec:
    """Judge model binding."""

    name: str
    backend: str
    model: str


JUDGE_SYSTEM_PROMPT = (
    "You are a strict legal QA evaluator. Score 1-5 and return JSON only with keys: "
    "legal_accuracy, summary_quality, risk_identification_quality, clause_extraction_quality, "
    "hallucination_risk, executive_usefulness, overall."
)


def _judge_prompt(input_text: str, reference_json: dict, prediction_json: dict) -> str:
    return (
        "Evaluate prediction against reference for legal quality.\n"
        "Use 1-5 integer scores where 5 is best, except hallucination_risk where 5 means high risk.\n\n"
        f"Input text:\n{input_text[:1200]}\n\n"
        f"Reference:\n{reference_json}\n\n"
        f"Prediction:\n{prediction_json}\n\n"
        "Return JSON only."
    )


def run_judge(
    spec: JudgeSpec,
    input_text: str,
    reference_json: dict,
    prediction_json: dict,
    generation_cfg: GenerationConfig,
) -> dict:
    """Run one judge model and parse score JSON."""

    client = UnifiedModelClient(backend=spec.backend, model=spec.model)
    prompt = _judge_prompt(input_text=input_text, reference_json=reference_json, prediction_json=prediction_json)
    raw = client.generate(prompt=prompt, system_prompt=JUDGE_SYSTEM_PROMPT, config=generation_cfg)
    parsed = try_parse_json(raw) or {}

    defaults = {
        "legal_accuracy": 3,
        "summary_quality": 3,
        "risk_identification_quality": 3,
        "clause_extraction_quality": 3,
        "hallucination_risk": 3,
        "executive_usefulness": 3,
        "overall": 3,
    }
    for key, value in defaults.items():
        parsed[key] = int(parsed.get(key, value))

    return parsed
