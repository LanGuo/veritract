import pytest
from veritract.optimizer import _score_results, _mutate_prompt
from veritract.llm import MockLLM


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


def test_mutate_prompt_returns_string():
    llm = MockLLM()
    llm.register("Improve the extraction prompt", {"text": "New improved prompt text."})
    result = _mutate_prompt(
        current_prompt="Extract drug and sample_size.",
        schema={"type": "object", "properties": {"drug": {"type": "string"}, "sample_size": {"type": "string"}}, "required": ["drug", "sample_size"]},
        failures=[{"field": "drug", "extracted": "aspirin", "expected": "metformin 500mg"}],
        llm=llm,
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_mutate_prompt_falls_back_on_llm_error():
    llm = MockLLM()
    # No registered response → MockLLM raises ValueError → should fall back to current_prompt
    result = _mutate_prompt(
        current_prompt="My prompt.",
        schema={"type": "object", "properties": {"drug": {"type": "string"}}, "required": ["drug"]},
        failures=[],
        llm=llm,
    )
    assert result == "My prompt."


from veritract.optimizer import optimize_prompt

_SOURCE = "The trial enrolled 248 patients. Patients received metformin 500mg twice daily."
_OPT_SCHEMA = {
    "type": "object",
    "properties": {"drug": {"type": "string"}, "sample_size": {"type": "string"}},
    "required": ["drug", "sample_size"],
}
_EXAMPLES = [
    {
        "text": _SOURCE,
        "fields": {"drug": "metformin 500mg twice daily", "sample_size": "248 patients"},
    }
]


def test_optimize_prompt_returns_string():
    llm = MockLLM()
    # MockLLM matches "248 patients" substring → used for extract() calls
    llm.register("248 patients", {"drug": "metformin 500mg twice daily", "sample_size": "248 patients"})
    # MockLLM matches "Improve" → used for _mutate_prompt calls
    llm.register("Improve", {"text": "Better prompt."})
    result = optimize_prompt(_EXAMPLES, _OPT_SCHEMA, llm, n_iter=1)
    assert isinstance(result, str)
    assert len(result) > 0


def test_optimize_prompt_supervised_with_gt():
    llm = MockLLM()
    llm.register("248 patients", {"drug": "metformin 500mg twice daily", "sample_size": "248 patients"})
    llm.register("Improve", {"text": "Refined prompt."})
    result = optimize_prompt(
        _EXAMPLES, _OPT_SCHEMA, llm, n_iter=2,
        ground_truth=[{"drug": "metformin 500mg twice daily", "sample_size": "248 patients"}],
    )
    assert isinstance(result, str)


def test_optimize_prompt_zero_iter_returns_initial():
    llm = MockLLM()
    result = optimize_prompt(_EXAMPLES, _OPT_SCHEMA, llm, n_iter=0)
    assert isinstance(result, str)


def test_optimize_prompt_no_examples_raises():
    llm = MockLLM()
    with pytest.raises(ValueError, match="examples"):
        optimize_prompt([], _OPT_SCHEMA, llm, n_iter=1)


def test_optimize_prompt_importable_from_package():
    from veritract import optimize_prompt as op
    assert callable(op)
