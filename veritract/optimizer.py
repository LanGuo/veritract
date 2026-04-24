from __future__ import annotations

import json as _json
import random as _random
from veritract.types import GroundedField, QuarantinedField
from veritract.extraction import extract, _build_prompt


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


def optimize_prompt(
    examples: list[dict],
    schema: dict,
    llm,
    *,
    n_iter: int = 3,
    n_sample: int | None = None,
    ground_truth: list[dict] | None = None,
    seed: int | None = None,
) -> str:
    """Iteratively refine an extraction prompt using extraction outcomes as feedback.

    Args:
        examples: List of dicts with keys "text" and "fields". Used as few-shot
            context in the initial prompt; "fields" values used as GT when
            ground_truth is None.
        schema: JSON Schema dict defining extraction fields.
        llm: LLM client with a chat(messages, schema=None) method.
        n_iter: Number of refinement iterations. 0 returns the initial prompt.
        n_sample: Max examples to evaluate per iteration. Defaults to all.
        ground_truth: Optional list of GT dicts (parallel to examples) for
            supervised scoring. When None, unsupervised grounding rate is used.
        seed: Random seed for sampling reproducibility.

    Returns:
        The best-performing prompt string found across all iterations.

    Raises:
        ValueError: If examples is empty.
    """
    if not examples:
        raise ValueError("examples must be non-empty")

    rng = _random.Random(seed)
    fields = list(schema.get("properties", {}).keys())

    current_prompt = _build_prompt(examples[0]["text"], schema, prompt=None, examples=examples)

    best_prompt = current_prompt
    best_score = -1.0

    for iteration in range(n_iter):
        indices = list(range(len(examples)))
        if n_sample is not None and n_sample < len(indices):
            indices = rng.sample(indices, n_sample)

        scores = []
        failures: list[dict] = []

        for idx in indices:
            ex = examples[idx]
            source = ex["text"]
            gt = ground_truth[idx] if ground_truth is not None else None

            eval_prompt = current_prompt + f"\n\nText:\n{source[:6000]}"

            try:
                result = extract(source, schema, llm, prompt=eval_prompt, mode="fuzzy")
            except Exception:
                scores.append(0.0)
                continue

            score = _score_results(result.extracted, result.quarantined, gt)
            scores.append(score)

            if gt is not None:
                for field in fields:
                    gf = result.extracted.get(field)
                    extracted_val = gf["value"] if gf else ""
                    expected_val = gt.get(field, "")
                    ev = extracted_val.strip().lower()
                    eg = expected_val.strip().lower()
                    if not (ev and eg and (ev in eg or eg in ev)):
                        failures.append({
                            "field": field,
                            "extracted": extracted_val,
                            "expected": expected_val,
                        })
            else:
                for qf in result.quarantined:
                    failures.append({
                        "field": qf["field_name"],
                        "extracted": qf["value"],
                        "reason": qf["reason"],
                    })

        mean_score = sum(scores) / len(scores) if scores else 0.0

        if mean_score > best_score:
            best_score = mean_score
            best_prompt = current_prompt

        if iteration < n_iter - 1:
            current_prompt = _mutate_prompt(current_prompt, schema, failures, llm)

    return best_prompt
