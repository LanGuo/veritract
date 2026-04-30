from __future__ import annotations

import base64
import re
from veritract.types import Span, GroundedField, QuarantinedField, ExtractionResult
from veritract.grounding import ExtractionGrounder

# Matches at least two consecutive word characters (letters, digits, _).
# Values that don't contain this are GBNF artifacts (pure punctuation/whitespace).
_HAS_WORD = re.compile(r"\w{2,}")
# Strip leading/trailing non-word noise that GBNF sometimes prepends/appends.
_STRIP_NOISE = re.compile(r'^[\s\W]+|[\s\W]+$')
# Model outputs these when a field is absent instead of returning "".
_EMPTY_SENTINEL = re.compile(
    r'^(empty\s+string|not\s+(found|provided|available|mentioned)|n/?a|none|null)$',
    re.IGNORECASE,
)

_LLM_GROUND_CONFIDENCE = 80.0

_LLM_GROUND_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "span": {"type": "string"},
    },
    "required": ["supported", "span"],
}


def _build_prompt(
    text: str,
    schema: dict,
    prompt: str | None,
    examples: list[dict] | None = None,
    max_text_chars: int | None = 6000,
) -> str:
    body = text if max_text_chars is None else text[:max_text_chars]
    if prompt is not None:
        return f"{prompt}\n\nText:\n{body}"
    fields = list(schema.get("properties", {}).keys())
    fields_str = ", ".join(fields)
    parts = [
        f"Extract the following fields from the text below.\n"
        f"Rules:\n"
        f"- Copy the exact verbatim phrase from the text. Do not paraphrase, abbreviate, or synthesise.\n"
        f'- If a field is not present in the text, return "".\n'
        f"Return JSON with exactly these fields: {fields_str}.",
    ]
    if examples:
        import json as _json
        parts.append("\nExamples:")
        for ex in examples:
            ex_text = ex.get("text", "")
            ex_fields = {f: ex.get("fields", {}).get(f, "") for f in fields}
            parts.append(f"\nText:\n{ex_text}\nOutput: {_json.dumps(ex_fields)}")
        parts.append("\nNow extract from the text below.")
    parts.append(f"\nText:\n{body}")
    return "\n".join(parts)


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
        if _EMPTY_SENTINEL.match(cleaned):
            continue  # model said "empty string" / "N/A" / etc — treat as absent
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


def extract_raw(
    text: str,
    schema: dict,
    llm,
    *,
    prompt: str | None = None,
    examples: list[dict] | None = None,
    images: list[str] | None = None,
    doc_id: str | None = None,
    source_type: str = "text",
    max_text_chars: int | None = 6000,
) -> "RawExtractionResult":
    """Call the LLM and return sanitized field values without source verification.

    Use ground() on the returned RawExtractionResult to apply verification.
    This split lets you reuse the same LLM output across multiple grounding
    strategies or save raw outputs for offline analysis.
    """
    from veritract.types import RawExtractionResult
    content = _build_prompt(text, schema, prompt, examples, max_text_chars)
    message: dict = {"role": "user", "content": content}
    if images:
        message["images"] = images
    raw = llm.chat([message], schema=schema)
    raw_strings, garbage = _sanitize_raw_values(raw)
    return RawExtractionResult(
        fields=raw_strings,
        garbage=garbage,
        source_text=text,
        doc_id=doc_id,
        source_type=source_type,
    )


def ground(
    raw: "RawExtractionResult",
    llm=None,
    *,
    mode: str = "full",
    thresholds: dict[str, int] | None = None,
) -> ExtractionResult:
    """Apply source verification to a RawExtractionResult.

    Args:
        raw: Output of extract_raw().
        llm: Required only for mode="full" (LLM re-verification of quarantined
            fields). Pass None for mode="fuzzy" or mode="no-grounding".
        mode: Same semantics as extract(). "full" requires llm.
        thresholds: Custom source_type → fuzzy threshold mapping.

    Raises:
        ValueError: If mode is unrecognised.
    """
    if mode not in _VERIFICATION_MODES:
        raise ValueError(f"mode must be one of {_VERIFICATION_MODES!r}, got {mode!r}")

    if mode == "no-grounding":
        extracted = {
            k: GroundedField(value=v, span=None, confidence=100.0)
            for k, v in raw.fields.items()
        }
        return ExtractionResult(extracted=extracted, quarantined=raw.garbage)

    grounder = ExtractionGrounder(thresholds=thresholds)
    grounded, quarantined = grounder.ground_extracted_data(
        raw.fields,
        source_text=raw.source_text,
        doc_id=raw.doc_id,
        source_type=raw.source_type,
    )
    quarantined = raw.garbage + quarantined

    if mode == "full" and quarantined:
        promoted, still_quarantined = _auto_llm_ground(
            quarantined, raw.source_text, llm, raw.doc_id, raw.source_type
        )
        grounded.update(promoted)
        quarantined = still_quarantined

    return ExtractionResult(extracted=grounded, quarantined=quarantined)


_VERIFICATION_MODES = ("full", "fuzzy", "no-grounding")


def extract(
    text: str,
    schema: dict,
    llm,
    *,
    mode: str = "full",
    prompt: str | None = None,
    examples: list[dict] | None = None,
    images: list[str] | None = None,
    doc_id: str | None = None,
    source_type: str = "text",
    grounding: bool | None = None,
    auto_reground: bool | None = None,
    thresholds: dict[str, int] | None = None,
) -> ExtractionResult:
    """Extract structured fields from text using an LLM with source verification.

    Args:
        text: Source text to extract from.
        schema: JSON Schema dict defining fields. Passed to LLM as constrained
            output format (Ollama ``format=`` param for GBNF constrained decoding).
        llm: Any object with a ``chat(messages, schema=None, think=False)`` method.
            Use ``LLMClient`` for Ollama or ``MockLLM`` for tests.
        mode: Verification level. Controls the latency/reliability tradeoff:

            ``"full"`` *(default)* — fuzzy grounding + LLM re-verification of
            quarantined fields. Highest reliability; ~1 extra LLM call per
            quarantined field (typically 10–15% of fields).

            ``"fuzzy"`` — fuzzy grounding only, no LLM re-verification.
            Quarantined fields remain quarantined. ~30–40% lower latency than
            ``"full"`` when quarantine rate is non-zero.

            ``"no-grounding"`` — raw LLM output accepted as-is, no source
            verification. Fastest; no quarantine. Use when latency dominates
            and downstream consumers tolerate unverified values.

            The legacy ``grounding`` and ``auto_reground`` boolean parameters
            override ``mode`` when explicitly passed.

        prompt: Custom extraction prompt. If None, a prompt is generated from schema
            field names and the source text (truncated to 6000 chars).
            If provided, the prompt is used verbatim and the caller is responsible
            for: (1) embedding the source ``text`` in the prompt body,
            (2) instructing the model to return JSON with exactly the field names
            defined in ``schema``, and (3) NOT including instructions that conflict
            with verbatim extraction (grounding will still verify against ``text``).
            The ``examples`` parameter is ignored when ``prompt`` is provided.
        examples: Optional few-shot examples to embed in the auto-generated prompt.
            Each entry is a dict with keys ``"text"`` (source passage) and
            ``"fields"`` (dict mapping schema field names to expected verbatim
            values from that passage). Ignored when ``prompt`` is provided.
            Example::

                examples=[{
                    "text": "248 patients received metformin 500mg for 12 months.",
                    "fields": {"drug": "metformin 500mg", "sample_size": "248 patients"},
                }]
        images: Optional list of base64-encoded PNG/JPEG strings for multimodal extraction.
        doc_id: Optional document identifier stored in provenance spans.
        source_type: Label for the source type, used for adaptive grounding thresholds.
            Defaults to "text". Use "abstract" (threshold 75) or "fulltext" (threshold 85).
        grounding: Deprecated — use ``mode`` instead. If explicitly passed, overrides
            the grounding step implied by ``mode``.
        auto_reground: Deprecated — use ``mode`` instead. If explicitly passed,
            overrides the LLM re-verification step implied by ``mode``.
        thresholds: Custom source_type → threshold mapping, overrides built-in defaults.

    Returns:
        ExtractionResult with .extracted (grounded fields), .quarantined, and .provenance.

    Raises:
        ValueError: If ``mode`` is not one of the recognised verification levels.
    """
    if mode not in _VERIFICATION_MODES:
        raise ValueError(f"mode must be one of {_VERIFICATION_MODES!r}, got {mode!r}")

    # Resolve legacy boolean flags into effective mode.
    _mode_grounding = mode != "no-grounding"
    _mode_reground = mode == "full"
    _do_grounding = grounding if grounding is not None else _mode_grounding
    _do_reground = auto_reground if auto_reground is not None else _mode_reground
    if not _do_grounding:
        effective_mode = "no-grounding"
    elif not _do_reground:
        effective_mode = "fuzzy"
    else:
        effective_mode = "full"

    raw = extract_raw(
        text, schema, llm,
        prompt=prompt, examples=examples, images=images,
        doc_id=doc_id, source_type=source_type,
    )
    return ground(raw, llm, mode=effective_mode, thresholds=thresholds)
