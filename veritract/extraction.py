from __future__ import annotations

import base64
import re
from veritract.types import Span, GroundedField, QuarantinedField, ExtractionResult
from veritract.grounding import ExtractionGrounder

# Matches at least two consecutive word characters (letters, digits, _).
# Values that don't contain this are GBNF artifacts (pure punctuation/whitespace).
_HAS_WORD = re.compile(r"\w{2,}")
# Strip leading/trailing non-word noise that GBNF sometimes prepends/appends.
_STRIP_NOISE = re.compile(r'^[\s\W]+|[\s,.:;"\']+$')

_LLM_GROUND_CONFIDENCE = 80.0

_LLM_GROUND_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "span": {"type": "string"},
    },
    "required": ["supported", "span"],
}


def _build_prompt(text: str, schema: dict, prompt: str | None) -> str:
    if prompt is not None:
        return prompt
    fields = list(schema.get("properties", {}).keys())
    fields_str = ", ".join(fields)
    return (
        f"Extract the following fields from the text below. "
        f"Return JSON with exactly these fields: {fields_str}.\n\n"
        f"Text:\n{text[:6000]}"
    )


def _locate_span(
    span_text: str,
    source_text: str,
    doc_id: str | None,
    source_type: str,
) -> Span | None:
    if not span_text:
        return None
    idx = source_text.lower().find(span_text.lower())
    if idx == -1:
        return None
    return Span(
        doc_id=doc_id,
        source_type=source_type,
        char_start=idx,
        char_end=idx + len(span_text),
        text=source_text[idx: idx + len(span_text)],
        provenance_type="inferred",
    )


def _auto_llm_ground(
    quarantined: list[QuarantinedField],
    source_text: str,
    llm,
    doc_id: str | None,
    source_type: str,
) -> tuple[dict[str, GroundedField], list[QuarantinedField]]:
    """Re-verify each quarantined field with a targeted LLM call.

    Promoted fields get provenance_type="inferred" and a located span when
    the LLM returns a verbatim quote that can be found in the source.
    """
    promoted: dict[str, GroundedField] = {}
    remaining: list[QuarantinedField] = []

    for qf in quarantined:
        field_name = qf["field_name"]
        value = qf["value"]
        try:
            result = llm.chat([{
                "role": "user",
                "content": (
                    f"Verify whether the extracted value is supported by the source text, "
                    f"even if phrased differently (paraphrased, abbreviated, or expressed differently).\n\n"
                    f"Field: {field_name}\n"
                    f"Extracted value: \"{value}\"\n\n"
                    f"Source text:\n{source_text[:6000]}\n\n"
                    "Return JSON: {\"supported\": true or false, "
                    "\"span\": \"verbatim quote from source that supports this, or empty string\"}"
                ),
            }], schema=_LLM_GROUND_SCHEMA)

            if result.get("supported"):
                span = _locate_span(result.get("span", ""), source_text, doc_id, source_type)
                promoted[field_name] = GroundedField(
                    value=value,
                    span=span,
                    confidence=_LLM_GROUND_CONFIDENCE,
                )
            else:
                remaining.append(qf)
        except Exception:
            remaining.append(qf)

    return promoted, remaining


def _sanitize_raw_values(
    raw: dict,
) -> tuple[dict[str, str], list[QuarantinedField]]:
    """Strip noise from GBNF-decoded strings; quarantine values with no word content.

    GBNF constrained decoding guarantees syntactically valid JSON but can produce
    strings that are pure punctuation (e.g. \"','\") or have leading JSON artifacts
    (e.g. \": 528\"). Strip the noise first; quarantine if nothing meaningful remains.
    """
    valid: dict[str, str] = {}
    garbage: list[QuarantinedField] = []
    for k, v in raw.items():
        if not isinstance(v, str):
            continue
        cleaned = _STRIP_NOISE.sub("", v)
        if _HAS_WORD.search(cleaned):
            valid[k] = cleaned
        else:
            garbage.append(QuarantinedField(
                field_name=k,
                value=v,
                reason=f"invalid extraction: no meaningful content in {v!r}",
            ))
    return valid, garbage


def load_images_b64(paths: list[str], max_images: int = 8) -> list[str]:
    """Read PNG/JPEG files and return base64-encoded strings for multimodal LLM calls."""
    images: list[str] = []
    for path in paths[:max_images]:
        try:
            with open(path, "rb") as fh:
                images.append(base64.b64encode(fh.read()).decode())
        except OSError:
            pass
    return images


def extract(
    text: str,
    schema: dict,
    llm,
    *,
    prompt: str | None = None,
    images: list[str] | None = None,
    doc_id: str | None = None,
    source_type: str = "text",
    grounding: bool = True,
    auto_reground: bool = True,
    thresholds: dict[str, int] | None = None,
) -> ExtractionResult:
    """Extract structured fields from text using an LLM with source verification.

    Args:
        text: Source text to extract from.
        schema: JSON Schema dict defining fields. Passed to LLM as constrained
            output format (Ollama ``format=`` param for GBNF constrained decoding).
        llm: Any object with a ``chat(messages, schema=None, think=False)`` method.
            Use ``LLMClient`` for Ollama or ``MockLLM`` for tests.
        prompt: Custom extraction prompt. If None, a prompt is generated from schema
            field names and the source text (truncated to 6000 chars). If provided,
            used verbatim — the caller is responsible for embedding the source text.
        images: Optional list of base64-encoded PNG/JPEG strings for multimodal extraction.
        doc_id: Optional document identifier stored in provenance spans.
        source_type: Label for the source type, used for adaptive grounding thresholds.
            Defaults to "text". Use "abstract" (threshold 75) or "fulltext" (threshold 85).
        grounding: If False, skip all grounding — all fields accepted as-is.
        auto_reground: If False, skip LLM re-verification of quarantined fields.
        thresholds: Custom source_type → threshold mapping, overrides built-in defaults.

    Returns:
        ExtractionResult with .extracted (grounded fields), .quarantined, and .provenance.
    """
    content = _build_prompt(text, schema, prompt)
    message: dict = {"role": "user", "content": content}
    if images:
        message["images"] = images

    raw = llm.chat([message], schema=schema)
    raw_strings, garbage_fields = _sanitize_raw_values(raw)

    if not grounding:
        extracted = {
            k: GroundedField(value=v, span=None, confidence=100.0)
            for k, v in raw_strings.items()
        }
        return ExtractionResult(extracted=extracted, quarantined=garbage_fields)

    grounder = ExtractionGrounder(thresholds=thresholds)
    grounded, quarantined_fields = grounder.ground_extracted_data(
        raw_strings,
        source_text=text,
        doc_id=doc_id,
        source_type=source_type,
    )

    quarantined_fields = garbage_fields + quarantined_fields

    if auto_reground and quarantined_fields:
        promoted, still_quarantined = _auto_llm_ground(
            quarantined_fields, text, llm, doc_id, source_type
        )
        grounded.update(promoted)
        quarantined_fields = still_quarantined

    return ExtractionResult(extracted=grounded, quarantined=quarantined_fields)
