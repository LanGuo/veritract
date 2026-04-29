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
