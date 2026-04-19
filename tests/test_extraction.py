import pytest
from veritract.extraction import extract, _build_prompt, _locate_span
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
