import pytest
from veritract.optimizer import _score_results


SCHEMA = {
    "type": "object",
    "properties": {
        "drug": {"type": "string"},
        "sample_size": {"type": "string"},
    },
    "required": ["drug", "sample_size"],
}

SOURCE = "Patients received metformin 500mg. The trial enrolled 248 patients."


def _gf(value, span=None):
    return {"value": value, "span": span, "confidence": 90.0}


def test_score_supervised_all_correct():
    extracted = {"drug": _gf("metformin 500mg"), "sample_size": _gf("248 patients")}
    gt = {"drug": "metformin 500mg", "sample_size": "248 patients"}
    score = _score_results(extracted, [], gt)
    assert score == 1.0


def test_score_supervised_partial():
    extracted = {"drug": _gf("metformin 500mg")}
    gt = {"drug": "metformin 500mg", "sample_size": "248 patients"}
    score = _score_results(extracted, [], gt)
    # 1 correct out of 2 fields
    assert score == pytest.approx(0.5)


def test_score_supervised_quarantined_counts_wrong():
    extracted = {"drug": _gf("metformin 500mg")}
    quarantined = [{"field_name": "sample_size", "value": "bogus", "reason": "no match"}]
    gt = {"drug": "metformin 500mg", "sample_size": "248 patients"}
    score = _score_results(extracted, quarantined, gt)
    assert score == pytest.approx(0.5)


def test_score_unsupervised_grounding_rate():
    span = {"doc_id": None, "source_type": "text", "char_start": 0, "char_end": 5,
            "text": "hello", "provenance_type": "inferred"}
    extracted = {
        "drug": _gf("metformin 500mg", span=span),
        "sample_size": _gf("248 patients"),  # no span → ungrounded
    }
    score = _score_results(extracted, [], gt=None)
    # 1 grounded out of 2 fields
    assert score == pytest.approx(0.5)


def test_score_unsupervised_quarantine_hurts():
    quarantined = [{"field_name": "drug", "value": "x", "reason": "r"}]
    extracted = {}
    score = _score_results(extracted, quarantined, gt=None)
    # 0 grounded out of 1 total
    assert score == pytest.approx(0.0)
