"""Prepare supervised rows for causal language-model fine-tuning."""

from __future__ import annotations

import json

from legal_clause_analyzer.schemas import SupervisedRow


def format_for_sft(row: SupervisedRow) -> str:
    """Convert one supervised row to text-target format.

    The answer side is JSON so the model learns schema-constrained responses.
    """

    prompt = (
        "### Instruction\n"
        f"{row.instruction}\n\n"
        "### Input\n"
        f"{row.input_text}\n\n"
        "### Output JSON\n"
    )
    answer = json.dumps(row.output.model_dump(), ensure_ascii=False)
    return f"{prompt}{answer}"
