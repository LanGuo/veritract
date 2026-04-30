from __future__ import annotations

from pathlib import Path

from veritract.extraction import extract_raw, ground
from veritract.types import RawExtractionResult, QuarantinedField, ExtractionResult


def _phrase_score(value: str) -> int:
    """Lower score = preferred. Penalises multi-clause values and very long strings.

    Single verbatim phrases ("27B", "128K tokens", "GRPO") score 0.
    Sentences and paragraph fragments score higher and lose to compact values.
    """
    return value.count(". ") + value.count("\n") + (1 if len(value) > 200 else 0)

# Try to import DocumentConverter, but allow it to be None for lazy loading/mocking
try:
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentConverter = None  # type: ignore

_DOCLING_IMPORT_MSG = (
    "docling is required for PDF extraction: "
    "pip install 'veritract[pdf]'"
)


def _require_docling() -> None:
    try:
        import docling  # noqa: F401
    except ImportError:
        raise ImportError(_DOCLING_IMPORT_MSG)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int]]:
    if not text:
        return []
    chunks: list[tuple[str, int]] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append((text[start:end], start))
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _merge_raw_results(
    raw_results: list[RawExtractionResult],
    full_text: str,
    doc_id: str | None,
) -> RawExtractionResult:
    if not raw_results:
        return RawExtractionResult(
            fields={}, garbage=[], source_text=full_text,
            doc_id=doc_id, source_type="pdf",
        )
    merged_fields: dict[str, str] = {}
    merged_garbage: list[QuarantinedField] = []
    for raw in raw_results:
        for field, value in raw.fields.items():
            if not value:
                continue
            current = merged_fields.get(field, "")
            if not current:
                merged_fields[field] = value
            elif _phrase_score(value) < _phrase_score(current):
                merged_fields[field] = value  # fewer clause breaks wins
            elif _phrase_score(value) == _phrase_score(current) and len(value) > len(current):
                merged_fields[field] = value  # same score: longer wins
        merged_garbage.extend(raw.garbage)
    return RawExtractionResult(
        fields=merged_fields,
        garbage=merged_garbage,
        source_text=full_text,
        doc_id=doc_id,
        source_type="pdf",
    )


def extract_pdf(
    path,
    schema: dict,
    llm,
    *,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    mode: str = "full",
    prompt: str | None = None,
    examples: list | None = None,
    thresholds: dict | None = None,
) -> ExtractionResult:
    """Extract structured fields from a PDF file using docling for PDF→markdown conversion.

    Requires: pip install 'veritract[pdf]'
    """
    if DocumentConverter is None:
        _require_docling()

    doc_id = Path(path).name
    converter = DocumentConverter()
    doc_result = converter.convert(str(path))
    full_text = doc_result.document.export_to_markdown()

    raw_results = []
    for chunk_text, _ in _chunk_text(full_text, chunk_size, chunk_overlap):
        raw = extract_raw(
            chunk_text,
            schema,
            llm,
            prompt=prompt,
            examples=examples,
            doc_id=doc_id,
            source_type="pdf",
            max_text_chars=None,  # chunk_size already controls window; don't truncate again
        )
        raw_results.append(raw)

    merged = _merge_raw_results(raw_results, full_text=full_text, doc_id=doc_id)
    return ground(merged, llm, mode=mode, thresholds=thresholds)
