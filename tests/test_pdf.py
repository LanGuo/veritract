# tests/test_pdf.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from veritract.types import RawExtractionResult, QuarantinedField


def _raw(fields: dict, source: str = "full doc") -> RawExtractionResult:
    return RawExtractionResult(
        fields=fields, garbage=[], source_text=source,
        doc_id="test.pdf", source_type="pdf",
    )


# --- _chunk_text ---

def test_chunk_text_single_chunk_when_text_fits():
    from veritract.pdf import _chunk_text
    result = _chunk_text("hello world", chunk_size=100, overlap=10)
    assert result == [("hello world", 0)]


def test_chunk_text_splits_on_size():
    from veritract.pdf import _chunk_text
    text = "a" * 50 + "b" * 50
    result = _chunk_text(text, chunk_size=50, overlap=0)
    assert len(result) == 2
    assert result[0] == ("a" * 50, 0)
    assert result[1] == ("b" * 50, 50)


def test_chunk_text_overlap_shifts_start():
    from veritract.pdf import _chunk_text
    text = "x" * 100
    result = _chunk_text(text, chunk_size=60, overlap=20)
    # Second chunk starts at 60-20=40
    assert result[1][1] == 40


def test_chunk_text_empty_string():
    from veritract.pdf import _chunk_text
    assert _chunk_text("", chunk_size=100, overlap=10) == []


# --- _merge_raw_results ---

def test_merge_picks_longest_value():
    from veritract.pdf import _merge_raw_results
    r1 = _raw({"drug": "aspirin"})
    r2 = _raw({"drug": "aspirin 100mg daily"})
    merged = _merge_raw_results([r1, r2], full_text="full", doc_id="f.pdf")
    assert merged.fields["drug"] == "aspirin 100mg daily"


def test_merge_skips_empty_values():
    from veritract.pdf import _merge_raw_results
    r1 = _raw({"drug": "", "sample_size": "100 patients"})
    r2 = _raw({"drug": "aspirin", "sample_size": ""})
    merged = _merge_raw_results([r1, r2], full_text="full", doc_id="f.pdf")
    assert merged.fields["drug"] == "aspirin"
    assert merged.fields["sample_size"] == "100 patients"


def test_merge_source_text_is_full_doc():
    from veritract.pdf import _merge_raw_results
    r = _raw({"drug": "aspirin"}, source="chunk text only")
    merged = _merge_raw_results([r], full_text="full document", doc_id="x.pdf")
    assert merged.source_text == "full document"


def test_merge_concatenates_garbage():
    from veritract.pdf import _merge_raw_results
    g1 = QuarantinedField(field_name="drug", value="?", reason="no content")
    g2 = QuarantinedField(field_name="outcome", value="!", reason="no content")
    r1 = RawExtractionResult(fields={}, garbage=[g1], source_text="", doc_id=None, source_type="pdf")
    r2 = RawExtractionResult(fields={}, garbage=[g2], source_text="", doc_id=None, source_type="pdf")
    merged = _merge_raw_results([r1, r2], full_text="", doc_id=None)
    assert len(merged.garbage) == 2


def test_merge_empty_list():
    from veritract.pdf import _merge_raw_results
    merged = _merge_raw_results([], full_text="doc", doc_id=None)
    assert merged.fields == {}
    assert merged.source_text == "doc"


def test_merge_prefers_compact_over_verbose():
    from veritract.pdf import _merge_raw_results
    # Sentence fragment (has ". ") loses to compact value (no ". ")
    r1 = _raw({"context_length": "We extend to 128K. This enables long-form reasoning."})
    r2 = _raw({"context_length": "128K"})
    merged = _merge_raw_results([r1, r2], full_text="full", doc_id="f.pdf")
    assert merged.fields["context_length"] == "128K"


def test_merge_same_score_longer_wins():
    from veritract.pdf import _merge_raw_results
    # Both single-phrase (score 0), same frequency: longer wins as tiebreak
    r1 = _raw({"drug": "aspirin"})
    r2 = _raw({"drug": "aspirin 100mg daily"})
    merged = _merge_raw_results([r1, r2], full_text="full", doc_id="f.pdf")
    assert merged.fields["drug"] == "aspirin 100mg daily"


def test_merge_frequency_beats_length():
    from veritract.pdf import _merge_raw_results
    # "36 trillion" appears in 3 chunks; "hundreds of billions" in 1 — frequency wins
    r1 = _raw({"pretraining_token_count": "36 trillion"})
    r2 = _raw({"pretraining_token_count": "hundreds of billions"})
    r3 = _raw({"pretraining_token_count": "36 trillion"})
    r4 = _raw({"pretraining_token_count": "36 trillion"})
    merged = _merge_raw_results([r1, r2, r3, r4], full_text="full", doc_id="f.pdf")
    assert merged.fields["pretraining_token_count"] == "36 trillion"


# --- extract_pdf ---

from unittest.mock import MagicMock, patch
from veritract.llm import MockLLM

SCHEMA = {
    "type": "object",
    "properties": {
        "drug": {"type": "string"},
        "sample_size": {"type": "string"},
        "outcome": {"type": "string"},
    },
    "required": ["drug", "sample_size", "outcome"],
}

FULL_MARKDOWN = (
    "A trial of **aspirin 100mg** enrolled 234 patients. "
    "Primary outcome was MI reduction at 12 months."
)


def _make_docling_mocks(markdown: str):
    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = markdown

    mock_converter = MagicMock()
    mock_converter.convert.return_value = mock_result

    return mock_converter


def test_extract_pdf_returns_extraction_result():
    from veritract.pdf import extract_pdf

    llm = MockLLM()
    llm.register("aspirin", {"drug": "aspirin 100mg", "sample_size": "234 patients", "outcome": "MI reduction"})

    with patch("veritract.pdf.DocumentConverter", return_value=_make_docling_mocks(FULL_MARKDOWN)):
        result = extract_pdf("fake.pdf", SCHEMA, llm, mode="no-grounding")

    assert result.extracted["drug"]["value"] == "aspirin 100mg"
    assert result.extracted["sample_size"]["value"] == "234 patients"
    assert result.extracted["outcome"]["value"] == "MI reduction"


def test_extract_pdf_grounds_against_full_text():
    from veritract.pdf import extract_pdf

    llm = MockLLM()
    llm.register("aspirin", {"drug": "aspirin 100mg", "sample_size": "234 patients", "outcome": "MI reduction"})

    with patch("veritract.pdf.DocumentConverter", return_value=_make_docling_mocks(FULL_MARKDOWN)):
        result = extract_pdf("fake.pdf", SCHEMA, llm, mode="fuzzy")

    assert result.extracted["sample_size"]["span"] is not None
    assert result.extracted["sample_size"]["span"]["provenance_type"] in ("direct", "paraphrased")


def test_extract_pdf_import_error_without_docling(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "docling" or name.startswith("docling."):
            raise ImportError("No module named 'docling'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    from veritract.pdf import _require_docling
    with pytest.raises(ImportError, match="veritract\\[pdf\\]"):
        _require_docling()


def test_extract_pdf_multi_chunk_merges_best_value():
    from veritract.pdf import extract_pdf

    long_markdown = "aspirin 100mg " * 300 + " | 234 patients enrolled"

    llm = MockLLM()
    # Register "234 patients" first so it matches before "aspirin" when both are present
    llm.register("234 patients", {"drug": "", "sample_size": "234 patients enrolled", "outcome": "MI reduction"})
    llm.register("aspirin", {"drug": "aspirin 100mg", "sample_size": "", "outcome": ""})

    with patch("veritract.pdf.DocumentConverter", return_value=_make_docling_mocks(long_markdown)):
        result = extract_pdf(
            "fake.pdf", SCHEMA, llm,
            mode="no-grounding", chunk_size=1000, chunk_overlap=50,
        )

    assert result.extracted["drug"]["value"] == "aspirin 100mg"
    assert result.extracted["sample_size"]["value"] == "234 patients enrolled"


def test_extract_pdf_importable_from_veritract():
    from veritract import extract_pdf
    assert callable(extract_pdf)
