from __future__ import annotations

import json as _json
from veritract.types import GroundedField, QuarantinedField


def _score_results(
    extracted: dict[str, GroundedField],
    quarantined: list[QuarantinedField],
    gt: dict[str, str] | None,
) -> float:
    """Return a 0-1 score for one extraction run.

    Supervised (gt provided): fraction of all GT fields whose extracted value
    case-insensitively contains or is contained by the ground-truth value.
    Quarantined fields count as wrong. Missing fields count as wrong.

    Unsupervised (gt=None): fraction of extracted fields that have a span
    (i.e. were grounded). Quarantined fields count as ungrounded.
    """
    all_fields = list(extracted.keys()) + [q["field_name"] for q in quarantined]
    if not all_fields and gt is None:
        return 0.0

    if gt is not None:
        gt_fields = set(gt.keys())
        correct = 0
        total = len(gt_fields)
        for field, gf in extracted.items():
            if field not in gt_fields:
                continue
            v = gf["value"].strip().lower()
            g = gt[field].strip().lower()
            if v and g and (v in g or g in v):
                correct += 1
        return correct / total if total > 0 else 0.0
    else:
        total = len(all_fields)
        grounded = sum(1 for gf in extracted.values() if gf["span"] is not None)
        return grounded / total if total > 0 else 0.0


_MUTATE_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}


def _mutate_prompt(
    current_prompt: str,
    schema: dict,
    failures: list[dict],
    llm,
) -> str:
    """Ask the LLM to suggest an improved extraction prompt.

    failures: list of dicts with keys 'field', 'extracted', 'expected' (or 'reason').
    Returns the new prompt string, or current_prompt if the LLM call fails.
    """
    fields = list(schema.get("properties", {}).keys())
    failure_text = ""
    if failures:
        lines = []
        for f in failures[:10]:
            lines.append(
                f"  field={f.get('field')!r}  extracted={f.get('extracted')!r}"
                f"  expected={f.get('expected', f.get('reason', '?'))!r}"
            )
        failure_text = "\n".join(lines)

    msg = (
        f"Improve the extraction prompt to fix these failures.\n\n"
        f"Current prompt:\n{current_prompt}\n\n"
        f"Schema fields: {fields}\n\n"
        f"Recent extraction failures (field / extracted value / expected value):\n"
        f"{failure_text or '(none — optimise for grounding)'}\n\n"
        "Rules the new prompt MUST follow:\n"
        "1. It must instruct the model to copy verbatim phrases from the source text.\n"
        "2. It must ask the model to return JSON with exactly these fields.\n"
        "3. It must embed the source text (the caller will append it).\n"
        'Respond with JSON: {"text": "<new prompt>"}'
    )
    try:
        result = llm.chat([{"role": "user", "content": msg}], schema=_MUTATE_SCHEMA)
        new_prompt = result.get("text", "").strip()
        return new_prompt if new_prompt else current_prompt
    except Exception:
        return current_prompt
