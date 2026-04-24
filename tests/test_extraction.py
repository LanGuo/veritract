import pytest
from veritract.extraction import extract, _build_prompt, _locate_span, _sanitize_raw_values
from veritract.llm import MockLLM


SOURCE = (
    "The randomized controlled trial enrolled 248 patients with type 2 diabetes. "
    "Participants received either metformin 500mg twice daily or placebo. "
    "The primary outcome was HbA1c reduction at 12 months. "
    "The metformin group showed a mean HbA1c reduction of 1.2% versus 0.3% in placebo."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "sample_size": {"type": "string"},
        "intervention": {"type": "string"},
        "primary_outcome": {"type": "string"},
    },
    "required": ["sample_size", "intervention", "primary_outcome"],
}


# --- _build_prompt ---

def test_build_prompt_uses_custom_prompt():
    custom = "My custom prompt."
    result = _build_prompt("hello", SCHEMA, custom)
    assert result == custom


def test_build_prompt_generates_from_schema():
    result = _build_prompt(SOURCE, SCHEMA, None)
    assert "sample_size" in result
    assert "intervention" in result
    assert "primary_outcome" in result
    assert SOURCE[:100] in result


def test_build_prompt_includes_examples():
    examples = [{
        "text": "100 patients received aspirin daily for 6 months.",
        "fields": {"sample_size": "100 patients", "intervention": "aspirin daily"},
    }]
    result = _build_prompt(SOURCE, SCHEMA, None, examples)
    assert "100 patients" in result
    assert "aspirin daily" in result
    assert SOURCE[:100] in result


def test_build_prompt_examples_ignored_when_prompt_provided():
    examples = [{"text": "irrelevant", "fields": {"sample_size": "irrelevant"}}]
    custom = "My custom prompt."
    result = _build_prompt("hello", SCHEMA, custom, examples)
    assert result == custom
    assert "irrelevant" not in result


# --- _locate_span ---

def test_locate_span_finds_verbatim_quote():
    span = _locate_span("248 patients", SOURCE, doc_id=None, source_type="text")
    assert span is not None
    assert span["provenance_type"] == "inferred"
    assert SOURCE[span["char_start"]:span["char_end"]].lower() == "248 patients"


def test_locate_span_returns_none_for_missing_quote():
    span = _locate_span("insulin glargine 10 units", SOURCE, doc_id=None, source_type="text")
    assert span is None


def test_locate_span_returns_none_for_empty_string():
    span = _locate_span("", SOURCE, doc_id=None, source_type="text")
    assert span is None


def test_locate_span_propagates_doc_id():
    span = _locate_span("248 patients", SOURCE, doc_id="doc42", source_type="abstract")
    assert span["doc_id"] == "doc42"
    assert span["source_type"] == "abstract"


# --- extract() ---

def test_extract_all_fields_grounded():
    llm = MockLLM()
    llm.register("sample_size", {
        "sample_size": "248 patients",
        "intervention": "metformin 500mg twice daily",
        "primary_outcome": "HbA1c reduction at 12 months",
    })
    result = extract(SOURCE, SCHEMA, llm)
    assert "sample_size" in result.extracted
    assert "intervention" in result.extracted
    assert "primary_outcome" in result.extracted
    assert result.quarantined == []


def test_extract_ungrounded_field_quarantined():
    llm = MockLLM()
    llm.register("sample_size", {
        "sample_size": "248 patients",
        "intervention": "insulin glargine 10 units nightly",
        "primary_outcome": "HbA1c reduction at 12 months",
    })
    result = extract(SOURCE, SCHEMA, llm, auto_reground=False)
    assert any(q["field_name"] == "intervention" for q in result.quarantined)


def test_extract_auto_reground_confirmed_field_promoted():
    llm = MockLLM()
    llm.register("sample_size", {
        "sample_size": "248 patients",
        "intervention": "metformin twice daily",
        "primary_outcome": "HbA1c reduction at 12 months",
    })
    # Re-verification confirms "metformin twice daily" with a verbatim span
    llm.register("metformin twice daily", {
        "supported": True,
        "span": "metformin 500mg twice daily",
    })
    result = extract(SOURCE, SCHEMA, llm, auto_reground=True)
    assert "intervention" in result.extracted


def test_extract_auto_reground_denied_field_quarantined():
    llm = MockLLM()
    llm.register("sample_size", {
        "sample_size": "248 patients",
        "intervention": "insulin glargine 10 units nightly",
        "primary_outcome": "HbA1c reduction at 12 months",
    })
    llm.register("insulin glargine", {"supported": False, "span": ""})
    result = extract(SOURCE, SCHEMA, llm, auto_reground=True)
    assert any(q["field_name"] == "intervention" for q in result.quarantined)


def test_extract_grounding_disabled():
    llm = MockLLM()
    llm.register("sample_size", {
        "sample_size": "completely made up value xyz",
        "intervention": "nonexistent drug",
        "primary_outcome": "fictional outcome",
    })
    result = extract(SOURCE, SCHEMA, llm, grounding=False)
    assert len(result.extracted) == 3
    assert result.quarantined == []


# --- mode parameter ---

def test_extract_mode_no_grounding_accepts_all():
    llm = MockLLM()
    llm.register("sample_size", {
        "sample_size": "completely made up value xyz",
        "intervention": "nonexistent drug",
        "primary_outcome": "fictional outcome",
    })
    result = extract(SOURCE, SCHEMA, llm, mode="no-grounding")
    assert len(result.extracted) == 3
    assert result.quarantined == []
    for f in result.extracted.values():
        assert f["span"] is None


def test_extract_mode_fuzzy_quarantines_without_regrounding():
    llm = MockLLM()
    llm.register("sample_size", {
        "sample_size": "248 patients",
        "intervention": "insulin glargine 10 units nightly",
        "primary_outcome": "HbA1c reduction at 12 months",
    })
    result = extract(SOURCE, SCHEMA, llm, mode="fuzzy")
    assert any(q["field_name"] == "intervention" for q in result.quarantined)


def test_extract_mode_full_attempts_regrounding():
    llm = MockLLM()
    llm.register("sample_size", {
        "sample_size": "248 patients",
        "intervention": "metformin twice daily",
        "primary_outcome": "HbA1c reduction at 12 months",
    })
    llm.register("metformin twice daily", {"supported": True, "span": "metformin 500mg twice daily"})
    result = extract(SOURCE, SCHEMA, llm, mode="full")
    assert "intervention" in result.extracted


def test_extract_mode_invalid_raises():
    llm = MockLLM()
    with pytest.raises(ValueError, match="mode must be one of"):
        extract(SOURCE, SCHEMA, llm, mode="turbo")


def test_extract_mode_explicit_bool_overrides_mode():
    """grounding=False overrides mode="full"."""
    llm = MockLLM()
    llm.register("sample_size", {
        "sample_size": "completely made up value xyz",
        "intervention": "nonexistent drug",
        "primary_outcome": "fictional outcome",
    })
    result = extract(SOURCE, SCHEMA, llm, mode="full", grounding=False)
    assert len(result.extracted) == 3
    assert result.quarantined == []


def test_extract_provenance_spans_present():
    llm = MockLLM()
    llm.register("sample_size", {
        "sample_size": "248 patients",
        "intervention": "metformin 500mg twice daily",
        "primary_outcome": "HbA1c reduction at 12 months",
    })
    result = extract(SOURCE, SCHEMA, llm)
    assert len(result.provenance) > 0
    for span in result.provenance:
        assert span["char_start"] >= 0
        assert span["char_end"] > span["char_start"]


def test_extract_doc_id_propagated():
    llm = MockLLM()
    llm.register("sample_size", {
        "sample_size": "248 patients",
        "intervention": "metformin 500mg twice daily",
        "primary_outcome": "HbA1c reduction at 12 months",
    })
    result = extract(SOURCE, SCHEMA, llm, doc_id="pmid:99999")
    for span in result.provenance:
        assert span["doc_id"] == "pmid:99999"


def test_extract_custom_prompt_used():
    llm = MockLLM()
    llm.register("CUSTOM_MARKER", {
        "sample_size": "248 patients",
        "intervention": "metformin 500mg twice daily",
        "primary_outcome": "HbA1c reduction at 12 months",
    })
    result = extract(
        SOURCE, SCHEMA, llm,
        prompt=f"CUSTOM_MARKER extract fields. Text: {SOURCE}",
    )
    assert "sample_size" in result.extracted


# --- _sanitize_raw_values ---

def test_sanitize_strips_leading_noise():
    valid, garbage = _sanitize_raw_values({"sample_size": ": 528 patients"})
    assert valid["sample_size"] == "528 patients"
    assert garbage == []


def test_sanitize_quarantines_punctuation_only():
    valid, garbage = _sanitize_raw_values({"sample_size": "','"})
    assert "sample_size" not in valid
    assert any(q["field_name"] == "sample_size" for q in garbage)


def test_sanitize_quarantines_empty_string():
    valid, garbage = _sanitize_raw_values({"field": ""})
    assert "field" not in valid
    assert any(q["field_name"] == "field" for q in garbage)


def test_sanitize_passes_normal_value():
    valid, garbage = _sanitize_raw_values({"drug": "metformin 500mg"})
    assert valid["drug"] == "metformin 500mg"
    assert garbage == []


def test_sanitize_skips_non_strings():
    valid, garbage = _sanitize_raw_values({"count": 42, "name": "valid"})
    assert "count" not in valid
    assert valid["name"] == "valid"


def test_extract_garbage_value_quarantined():
    llm = MockLLM()
    llm.register("sample_size", {
        "sample_size": "','" ,
        "intervention": "metformin 500mg twice daily",
        "primary_outcome": "HbA1c reduction at 12 months",
    })
    result = extract(SOURCE, SCHEMA, llm, auto_reground=False)
    assert "sample_size" not in result.extracted
    assert any(q["field_name"] == "sample_size" for q in result.quarantined)


# --- RawExtractionResult ---

from veritract.types import RawExtractionResult

def test_raw_extraction_result_fields():
    raw = RawExtractionResult(
        fields={"drug": "metformin", "sample_size": "248 patients"},
        garbage=[],
        source_text="248 patients received metformin.",
        doc_id="doc1",
        source_type="abstract",
    )
    assert raw.fields["drug"] == "metformin"
    assert raw.source_text == "248 patients received metformin."
    assert raw.doc_id == "doc1"
    assert raw.source_type == "abstract"
    assert raw.garbage == []


# --- extract_raw() ---

from veritract.extraction import extract_raw

def test_extract_raw_returns_raw_result():
    llm = MockLLM()
    llm.register("sample_size", {
        "sample_size": "248 patients",
        "intervention": "metformin 500mg twice daily",
        "primary_outcome": "HbA1c reduction at 12 months",
    })
    raw = extract_raw(SOURCE, SCHEMA, llm)
    assert isinstance(raw, RawExtractionResult)
    assert raw.fields["sample_size"] == "248 patients"
    assert raw.source_text == SOURCE
    assert raw.doc_id is None
    assert raw.source_type == "text"


def test_extract_raw_propagates_doc_id():
    llm = MockLLM()
    llm.register("sample_size", {
        "sample_size": "248 patients",
        "intervention": "metformin 500mg twice daily",
        "primary_outcome": "HbA1c reduction at 12 months",
    })
    raw = extract_raw(SOURCE, SCHEMA, llm, doc_id="pmid:1234", source_type="abstract")
    assert raw.doc_id == "pmid:1234"
    assert raw.source_type == "abstract"


def test_extract_raw_quarantines_garbage():
    llm = MockLLM()
    llm.register("sample_size", {
        "sample_size": "','",
        "intervention": "metformin 500mg twice daily",
        "primary_outcome": "HbA1c reduction at 12 months",
    })
    raw = extract_raw(SOURCE, SCHEMA, llm)
    assert "sample_size" not in raw.fields
    assert any(q["field_name"] == "sample_size" for q in raw.garbage)


# --- ground() ---

from veritract.extraction import ground


def _make_raw(fields, source=SOURCE):
    return RawExtractionResult(
        fields=fields,
        garbage=[],
        source_text=source,
        doc_id=None,
        source_type="text",
    )


def test_ground_no_grounding_mode_accepts_all():
    raw = _make_raw({"sample_size": "completely made up value xyz",
                     "intervention": "nonexistent drug"})
    result = ground(raw, llm=None, mode="no-grounding")
    assert len(result.extracted) == 2
    assert result.quarantined == []
    for f in result.extracted.values():
        assert f["span"] is None


def test_ground_fuzzy_quarantines_ungrounded():
    raw = _make_raw({"sample_size": "248 patients",
                     "intervention": "insulin glargine 10 units nightly"})
    result = ground(raw, llm=None, mode="fuzzy")
    assert "sample_size" in result.extracted
    assert any(q["field_name"] == "intervention" for q in result.quarantined)


def test_ground_full_promotes_via_llm():
    llm = MockLLM()
    llm.register("insulin glargine", {"supported": True, "span": "metformin 500mg twice daily"})
    raw = _make_raw({"sample_size": "248 patients",
                     "intervention": "insulin glargine 10 units nightly"})
    result = ground(raw, llm, mode="full")
    assert "intervention" in result.extracted


def test_ground_invalid_mode_raises():
    raw = _make_raw({"sample_size": "248 patients"})
    with pytest.raises(ValueError, match="mode must be one of"):
        ground(raw, llm=None, mode="turbo")


def test_extract_calls_extract_raw_and_ground():
    """extract() still works identically after refactor."""
    llm = MockLLM()
    llm.register("sample_size", {
        "sample_size": "248 patients",
        "intervention": "metformin 500mg twice daily",
        "primary_outcome": "HbA1c reduction at 12 months",
    })
    result = extract(SOURCE, SCHEMA, llm, mode="no-grounding")
    assert len(result.extracted) == 3
    assert result.quarantined == []
