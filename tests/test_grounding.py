from veritract.grounding import ExtractionGrounder
from veritract.types import GroundedField, QuarantinedField


SOURCE = (
    "The randomized controlled trial enrolled 248 patients with type 2 diabetes. "
    "Participants received either metformin 500mg twice daily or placebo. "
    "The primary outcome was HbA1c reduction at 12 months. "
    "The metformin group showed a mean HbA1c reduction of 1.2% versus 0.3% in placebo."
)


def test_exact_match_short_value():
    grounder = ExtractionGrounder()
    result = grounder.ground_field("sample_size", "248 patients", SOURCE, source_type="text")
    assert result is not None
    assert result["span"]["provenance_type"] == "direct"
    assert result["span"]["char_start"] >= 0
    assert SOURCE[result["span"]["char_start"]:result["span"]["char_end"]].lower() == "248 patients"


def test_exact_match_number_only():
    grounder = ExtractionGrounder()
    result = grounder.ground_field("sample_size", "248", SOURCE, source_type="text")
    assert result is not None
    assert result["span"]["provenance_type"] == "direct"


def test_fuzzy_match_paraphrase():
    grounder = ExtractionGrounder()
    result = grounder.ground_field(
        "primary_outcome",
        "HbA1c reduction at 12 months",
        SOURCE,
        source_type="text",
    )
    assert result is not None
    assert result["span"]["provenance_type"] in ("direct", "paraphrased")


def test_missing_value_quarantined():
    grounder = ExtractionGrounder()
    result = grounder.ground_field(
        "intervention", "insulin glargine 10 units nightly", SOURCE, source_type="text"
    )
    assert result is None


def test_doc_id_propagated_to_span():
    grounder = ExtractionGrounder()
    result = grounder.ground_field(
        "sample_size", "248", SOURCE, doc_id="pmid:12345678", source_type="abstract"
    )
    assert result is not None
    assert result["span"]["doc_id"] == "pmid:12345678"


def test_source_type_propagated_to_span():
    grounder = ExtractionGrounder()
    result = grounder.ground_field("sample_size", "248", SOURCE, source_type="fulltext")
    assert result is not None
    assert result["span"]["source_type"] == "fulltext"


def test_custom_threshold_strict_quarantines_fuzzy():
    grounder_strict = ExtractionGrounder(thresholds={"text": 99})
    # Value with tokens not meaningfully in source → high threshold rejects it
    value = "fasting glucose insulin resistance"
    result = grounder_strict.ground_field("x", value, SOURCE, source_type="text")
    assert result is None


def test_custom_threshold_lenient_passes_fuzzy():
    grounder_lenient = ExtractionGrounder(thresholds={"text": 10})
    # Same value passes with a very lenient threshold
    value = "fasting glucose insulin resistance"
    result = grounder_lenient.ground_field("x", value, SOURCE, source_type="text")
    assert result is not None


def test_ground_extracted_data_splits_grounded_and_quarantined():
    grounder = ExtractionGrounder()
    data = {
        "sample_size": "248",
        "intervention": "insulin glargine 10 units",
    }
    grounded, quarantined = grounder.ground_extracted_data(data, SOURCE, source_type="text")
    assert "sample_size" in grounded
    assert any(q["field_name"] == "intervention" for q in quarantined)


def test_ground_extracted_data_skips_empty_values():
    grounder = ExtractionGrounder()
    data = {"sample_size": "248", "missing_field": ""}
    grounded, quarantined = grounder.ground_extracted_data(data, SOURCE, source_type="text")
    assert "missing_field" not in grounded
    assert not any(q["field_name"] == "missing_field" for q in quarantined)


def test_ground_extracted_data_skips_non_string_values():
    grounder = ExtractionGrounder()
    data = {"sample_size": "248", "count": 42}  # type: ignore
    grounded, _ = grounder.ground_extracted_data(data, SOURCE, source_type="text")
    assert "count" not in grounded
