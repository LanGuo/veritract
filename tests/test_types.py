from veritract.types import Span, GroundedField, QuarantinedField, ExtractionResult


def test_import():
    import veritract  # noqa: F401


def test_span_fields():
    span = Span(
        doc_id="doc1",
        source_type="abstract",
        char_start=10,
        char_end=26,
        text="hello world test",
        provenance_type="direct",
    )
    assert span["char_end"] - span["char_start"] == len(span["text"])


def test_span_doc_id_optional():
    span = Span(
        doc_id=None,
        source_type="text",
        char_start=0,
        char_end=5,
        text="hello",
        provenance_type="paraphrased",
    )
    assert span["doc_id"] is None


def test_grounded_field():
    gf = GroundedField(value="42 patients", span=None, confidence=100.0)
    assert gf["value"] == "42 patients"
    assert gf["span"] is None


def test_quarantined_field():
    qf = QuarantinedField(
        field_name="sample_size",
        value="forty-two",
        reason="no matching span (score=30.0)",
    )
    assert qf["field_name"] == "sample_size"


def test_extraction_result_provenance():
    span = Span(
        doc_id=None, source_type="text",
        char_start=0, char_end=3, text="foo",
        provenance_type="direct",
    )
    result = ExtractionResult(
        extracted={
            "field_a": GroundedField(value="foo", span=span, confidence=100.0),
            "field_b": GroundedField(value="bar", span=None, confidence=80.0),
        },
        quarantined=[],
    )
    assert len(result.provenance) == 1
    assert result.provenance[0]["text"] == "foo"


def test_extraction_result_empty():
    result = ExtractionResult(extracted={}, quarantined=[])
    assert result.provenance == []


def test_public_api_importable():
    from veritract import extract, LLMClient, MockLLM, ExtractionResult
    from veritract import Span, GroundedField, QuarantinedField, load_images_b64
    assert callable(extract)
    assert callable(LLMClient)
    assert callable(MockLLM)
    assert callable(load_images_b64)
