"""Hallucination analysis for structured legal outputs."""

from __future__ import annotations

from legal_clause_analyzer.schemas import LegalAnalysis


def _norm_list(items: list[str]) -> set[str]:
    return {x.strip().lower() for x in items if x.strip()}


def hallucination_metrics(rows: list[dict]) -> dict[str, float]:
    """Measure unsupported and missing structured claims.

    Definitions:
    - unsupported claims: generated obligation/liability items not present in source text
      and not present in the reference output.
    - missing obligations/liabilities: reference items absent from prediction.
    """

    unsupported = 0
    predicted_total = 0
    missing_obligations = 0
    ref_obligations_total = 0
    missing_liabilities = 0
    ref_liabilities_total = 0

    for row in rows:
        source = row["input_text"].lower()
        ref = LegalAnalysis.model_validate(row["reference"])
        pred = LegalAnalysis.model_validate(row["prediction"])

        ref_ob = _norm_list(ref.obligations)
        ref_li = _norm_list(ref.liabilities)
        pred_ob = _norm_list(pred.obligations)
        pred_li = _norm_list(pred.liabilities)

        ref_obligations_total += len(ref_ob)
        ref_liabilities_total += len(ref_li)

        missing_obligations += len(ref_ob - pred_ob)
        missing_liabilities += len(ref_li - pred_li)

        for item in pred_ob | pred_li:
            predicted_total += 1
            if item and item not in source and item not in ref_ob and item not in ref_li:
                unsupported += 1

    unsupported_rate = unsupported / max(1, predicted_total)
    missing_obligation_rate = missing_obligations / max(1, ref_obligations_total)
    missing_liability_rate = missing_liabilities / max(1, ref_liabilities_total)

    return {
        "unsupported_claim_rate": unsupported_rate,
        "missing_obligation_rate": missing_obligation_rate,
        "missing_liability_rate": missing_liability_rate,
    }
