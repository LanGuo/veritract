from __future__ import annotations

import re
from typing import Any
from rapidfuzz import fuzz
from veritract.types import Span, GroundedField, QuarantinedField

_DEFAULT_THRESHOLDS: dict[str, int] = {"abstract": 75, "fulltext": 85}
_DEFAULT_THRESHOLD = 80
_SHORT_VALUE_CHARS = 20


def _ws_pattern(value: str) -> re.Pattern:
    """Regex that matches value with any whitespace (space/newline) between words.

    Handles the common case where the LLM normalises line-wrapped source text
    (e.g. "atorvastatin\n40 mg" → "atorvastatin 40 mg") before returning it.
    """
    return re.compile(r"\s+".join(re.escape(w) for w in value.split()), re.IGNORECASE)


def _find_exact(value: str, source_text: str) -> tuple[int, int] | None:
    """Return (start, end) for an exact or whitespace-flexible match, or None."""
    value_lower = value.lower()
    source_lower = source_text.lower()
    # Literal match (fastest path)
    if value_lower in source_lower:
        idx = source_lower.index(value_lower)
        return idx, idx + len(value)
    # Whitespace-flexible match (value tokens separated by any \s+)
    m = _ws_pattern(value).search(source_text)
    if m:
        return m.start(), m.end()
    return None


class ExtractionGrounder:
    """Grounds extracted field values against source text using fuzzy matching.

    Short values (<20 chars) use exact substring search.
    Longer values use rapidfuzz token_set_ratio with sentence-window localisation.
    Thresholds are source-type-adaptive: abstract=75, fulltext=85, others=80.
    """

    def __init__(
        self,
        thresholds: dict[str, int] | None = None,
        short_value_chars: int = _SHORT_VALUE_CHARS,
    ):
        self._thresholds = thresholds if thresholds is not None else _DEFAULT_THRESHOLDS
        self._short_value_chars = short_value_chars

    def _threshold_for(self, source_type: str) -> int:
        return self._thresholds.get(source_type, _DEFAULT_THRESHOLD)

    def ground_field(
        self,
        field_name: str,
        value: str,
        source_text: str,
        doc_id: str | None = None,
        source_type: str = "text",
    ) -> GroundedField | None:
        """Ground a single field value against source text.

        Returns a GroundedField on success, None if the value cannot be grounded.
        """
        threshold = self._threshold_for(source_type)
        value_lower = value.lower()
        source_lower = source_text.lower()

        # Try exact / whitespace-flexible match first — tightest possible span.
        exact = _find_exact(value, source_text)
        if exact:
            s, e = exact
            return GroundedField(
                value=value,
                span=Span(
                    doc_id=doc_id, source_type=source_type,
                    char_start=s, char_end=e,
                    text=source_text[s:e], provenance_type="direct",
                ),
                confidence=100.0,
            )

        # Short values that don't appear verbatim cannot be grounded by fuzzy.
        if len(value) < self._short_value_chars:
            return None

        # Longer values: fuzzy window search across sentence triples.
        overall_score = fuzz.token_set_ratio(value_lower, source_lower)
        if overall_score < threshold:
            return None

        sentences = re.split(r"(?<=[.!?])\s+", source_text)
        offsets: list[int] = []
        search_from = 0
        for s in sentences:
            idx = source_text.find(s, search_from)
            offsets.append(idx if idx != -1 else search_from)
            search_from = (idx + len(s)) if idx != -1 else search_from

        best_score = 0.0
        best_start = 0
        best_end = min(len(value) * 2, len(source_text))
        window_size = 3
        for i in range(len(sentences)):
            chunk = " ".join(sentences[i: i + window_size])
            score = fuzz.token_set_ratio(value_lower, chunk.lower())
            if score > best_score:
                best_score = score
                best_start = offsets[i]
                end_idx = min(i + window_size - 1, len(sentences) - 1)
                best_end = offsets[end_idx] + len(sentences[end_idx])

        # Tighten: try exact/ws-flexible match within the fuzzy window.
        window_exact = _find_exact(value, source_text[best_start:best_end])
        if window_exact:
            s, e = best_start + window_exact[0], best_start + window_exact[1]
            return GroundedField(
                value=value,
                span=Span(
                    doc_id=doc_id, source_type=source_type,
                    char_start=s, char_end=e,
                    text=source_text[s:e], provenance_type="direct",
                ),
                confidence=100.0,
            )

        return GroundedField(
            value=value,
            span=Span(
                doc_id=doc_id, source_type=source_type,
                char_start=best_start, char_end=best_end,
                text=source_text[best_start:best_end], provenance_type="paraphrased",
            ),
            confidence=overall_score,
        )

    def ground_extracted_data(
        self,
        extracted_data: dict[str, Any],
        source_text: str,
        doc_id: str | None = None,
        source_type: str = "text",
    ) -> tuple[dict[str, GroundedField], list[QuarantinedField]]:
        grounded: dict[str, GroundedField] = {}
        quarantined: list[QuarantinedField] = []

        for field_name, value in extracted_data.items():
            if not isinstance(value, str) or not value.strip():
                continue
            result = self.ground_field(field_name, value, source_text, doc_id, source_type)
            if result is not None:
                grounded[field_name] = result
            else:
                score = fuzz.token_set_ratio(value.lower(), source_text.lower())
                quarantined.append(QuarantinedField(
                    field_name=field_name,
                    value=value,
                    reason=f"no matching span (score={score:.1f})",
                ))

        return grounded, quarantined
