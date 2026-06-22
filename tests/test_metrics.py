from legal_clause_analyzer.evaluation.metrics import classification_metrics, extraction_metrics, summarization_metrics


def test_classification_metrics_range() -> None:
    scores = classification_metrics(["low", "high"], ["low", "medium"])
    assert 0.0 <= scores["f1"] <= 1.0


def test_extraction_metrics_range() -> None:
    scores = extraction_metrics([["a", "b"], ["c"]], [["a"], ["c", "d"]])
    assert 0.0 <= scores["precision"] <= 1.0
    assert 0.0 <= scores["recall"] <= 1.0


def test_summarization_metrics_keys() -> None:
    scores = summarization_metrics(["the contract is terminated"], ["contract terminated"])
    for key in ["rouge1", "rouge2", "rougeL", "bleu", "meteor", "bertscore_f1"]:
        assert key in scores
