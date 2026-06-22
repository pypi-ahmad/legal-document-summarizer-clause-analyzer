from legal_clause_analyzer.data.instruction_builder import row_to_supervision
from legal_clause_analyzer.data.lexglue_loader import RawExample


def test_row_to_supervision_structure() -> None:
    policy = {
        "risk_levels": {"low": [], "medium": ["taxes"], "high": ["indemnity"]},
        "liability_buckets": {"financial liability": ["indemnity"]},
        "red_flag_keywords": ["unlimited liability"],
        "obligation_keywords": ["shall"],
        "rights_keywords": ["may"],
        "restriction_keywords": ["must not"],
    }

    raw = RawExample(
        row_id="x",
        dataset="ledgar",
        split="train",
        text="Vendor shall indemnify client. Vendor may invoice quarterly.",
        labels=[0],
        label_names=["Indemnity"],
        metadata={},
    )

    out = row_to_supervision(raw, policy)
    assert out.output.executive_summary
    assert out.output.risk_level in {"low", "medium", "high"}
