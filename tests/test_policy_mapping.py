from pathlib import Path

from legal_clause_analyzer.data.policy_mapping import (
    infer_liabilities,
    infer_obligations,
    infer_red_flags,
    infer_risk_level,
    load_policy,
)


def test_risk_and_liability_mapping() -> None:
    policy = load_policy(Path("configs/policy_mapping.yaml"))
    text = "Supplier shall indemnify client and may be subject to sanctions for non-compliance."
    labels = ["Indemnity", "Sanctions"]

    risk = infer_risk_level(labels, text, policy)
    liabilities = infer_liabilities(labels, text, policy)

    assert risk == "high"
    assert "financial liability" in liabilities or "regulatory liability" in liabilities


def test_obligations_and_flags() -> None:
    policy = load_policy(Path("configs/policy_mapping.yaml"))
    text = "Provider shall maintain insurance. Terms allow unilateral termination without notice."
    labels = ["Insurances", "Termination"]

    obligations = infer_obligations(text, policy)
    red_flags = infer_red_flags(labels, text, policy)

    assert obligations
    assert any("unilateral" in x for x in red_flags)
