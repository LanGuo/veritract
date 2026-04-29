#!/usr/bin/env python3
"""Head-to-head benchmark: veritract vs LangExtract on clinical trial extraction.

Both packages use the same local Ollama model (gemma4:e4b by default).
The task is extracting structured fields from clinical trial abstracts.

Datasets:
  clinicaltrials — 50 real Phase III/IV trials from ClinicalTrials.gov (requires --build)
  ebmnlp        — 183 expert-annotated abstracts from EBM-NLP (Nye et al. 2018)
  synthetic     — 10 hand-crafted examples

Metrics:
  - Latency (seconds per sample)
  - Field accuracy (fuzzy match ≥70 vs ground truth)
  - Grounding rate (% of extracted fields with char-level provenance)
  - Quarantine rate (% of fields quarantined; veritract-only)

Usage:
  python benchmarks/benchmark.py [--model MODEL] [--dataset ebmnlp]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as _stats
import sys
import time
from pathlib import Path

from rapidfuzz import fuzz

# Add parent dir so the local veritract package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.dataset import SAMPLES as SYNTHETIC_SAMPLES, SCHEMA as SYNTHETIC_SCHEMA, FIELDS as SYNTHETIC_FIELDS  # noqa: E402
from benchmarks.clinicaltrials_dataset import get_samples as get_ct_samples, SCHEMA as CT_SCHEMA, FIELDS as CT_FIELDS  # noqa: E402
from benchmarks.ebmnlp_dataset import get_samples as get_ebmnlp_samples, SCHEMA as EBMNLP_SCHEMA, FIELDS as EBMNLP_FIELDS  # noqa: E402
from veritract import extract as vt_extract, extract_raw as vt_extract_raw, ground as vt_ground, LLMClient, optimize_prompt as vt_optimize_prompt, RawExtractionResult  # noqa: E402

_FUZZY_THRESHOLD = 70  # token_set_ratio threshold for "correct" answer

# ---------------------------------------------------------------------------
# Shared few-shot examples
# Used verbatim in both veritract (examples= param) and LangExtract (ExampleData).
# Chosen to demonstrate: complete drug name+dosage, written-out sample sizes,
# concise measure names (not result sentences).
# ---------------------------------------------------------------------------
_SHARED_EXAMPLES = [
    {
        "text": (
            "A double-blind randomized controlled trial enrolled 312 adults with "
            "rheumatoid arthritis. Patients received etanercept 50 mg subcutaneously "
            "once weekly or matching placebo for 24 weeks. The primary outcome was "
            "ACR20 response rate at week 24."
        ),
        "fields": {
            "drug": "etanercept 50 mg subcutaneously once weekly",
            "sample_size": "312 adults",
            "outcome": "ACR20 response rate",
            "duration": "24 weeks",
        },
    },
    {
        "text": (
            "Eighty-six patients with essential hypertension were randomly assigned "
            "to amlodipine 5 mg daily or hydrochlorothiazide 12.5 mg daily. The primary "
            "endpoint was change in 24-hour ambulatory systolic blood pressure from "
            "baseline at 12 weeks."
        ),
        "fields": {
            "drug": "amlodipine 5 mg daily",
            "sample_size": "Eighty-six patients",
            "outcome": "change in 24-hour ambulatory systolic blood pressure from baseline",
            "duration": "12 weeks",
        },
    },
]

# Fields excluded from the primary accuracy metric with reasons:
#   duration  — CT.gov timeFrame phrasing rarely matches abstract phrasing
#               (e.g. "from randomisation until death" vs "28-day cycles")
# EBM-NLP has no duration field so UNSCORED_FIELDS only applies to clinicaltrials.
UNSCORED_FIELDS: dict[str, set[str]] = {
    "clinicaltrials": {"duration"},
    "ebmnlp": set(),
    "synthetic": set(),
}

_T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
           6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 14: 2.145, 29: 2.045}


def _ci95(values: list[float]) -> tuple[float, float]:
    """Return (mean, half-width of 95% CI using t-distribution). Half-width is 0 for N=1."""
    n = len(values)
    if n <= 1:
        return (values[0] if values else 0.0), 0.0
    m = _stats.mean(values)
    s = _stats.stdev(values)
    t = _T_CRIT.get(n - 1, 1.96)
    return m, t * s / math.sqrt(n)


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}},
    "required": ["correct"],
}

# Fields where fuzzy matching is unreliable and LLM judging helps most.
_LLM_JUDGE_FIELDS = {"outcome", "drug"}


def _is_verbatim(value: str, source: str) -> bool:
    """Return True if value appears as a case-insensitive substring of source."""
    v = value.strip()
    return bool(v) and v.lower() in source.lower()


def _extract_number(s: str) -> int | None:
    """Return the first integer found in s, or None."""
    m = re.search(r"\d+", s.replace(",", ""))
    return int(m.group()) if m else None


def _is_abbrev_of(short: str, long: str) -> bool:
    """Return True if short is an acronym/initialism of long.

    Handles cases like 'OC' matching 'Oral contraceptive'.
    """
    short = short.strip()
    if not short.isupper() or len(short) < 2:
        return False
    long_words = long.split()
    if len(short) > len(long_words):
        return False
    return short == "".join(w[0].upper() for w in long_words[: len(short)])


def _fuzzy_match(predicted: str, expected: str) -> bool:
    """Multi-strategy fuzzy match: token_set_ratio + partial_ratio + acronym."""
    p, e = predicted.lower().strip(), expected.lower().strip()
    if not p or not e:
        return False
    if fuzz.token_set_ratio(p, e) >= _FUZZY_THRESHOLD:
        return True
    # Partial containment: one is a meaningful substring of the other (≥85%).
    if fuzz.partial_ratio(p, e) >= 85:
        return True
    # Acronym: "OC" → "Oral contraceptive", checked in both directions.
    p_words = predicted.split()
    e_words = expected.split()
    for token in p_words:
        if _is_abbrev_of(token, expected):
            return True
    for token in e_words:
        if _is_abbrev_of(token, predicted):
            return True
    return False


def _score_field(field: str, predicted: str, expected: str | list[str], dataset: str = "clinicaltrials") -> bool:
    """Field-specific scoring logic. expected may be a single string or a list of GT spans.

    When a list is given, returns True if predicted matches ANY span (deduped).
    """
    if isinstance(expected, list):
        seen: set[str] = set()
        for e in expected:
            if e not in seen:
                seen.add(e)
                if _score_field(field, predicted, e, dataset):
                    return True
        return False
    if field == "sample_size":
        # Try numeric proximity first (handles CT.gov enrolled vs analyzed N,
        # and EBM-NLP written-out numbers when LLM normalizes to digits).
        gt_n = _extract_number(expected)
        ex_n = _extract_number(predicted)
        if gt_n and ex_n and abs(gt_n - ex_n) / gt_n <= 0.15:
            return True
        # Fall back to fuzzy (handles written-out numbers like "Two hundred...")
    return _fuzzy_match(predicted, expected)


def _llm_judge(field: str, predicted: str, expected: str | list[str], source_text: str, llm) -> bool:
    """Semantic equivalence judge via LLM with a clinical extraction rubric.

    expected may be a single GT span or a list of all valid GT spans for the field.
    Used for fields (outcome, drug) where fuzzy string matching is unreliable.
    """
    if not predicted.strip():
        return False
    if isinstance(expected, list):
        seen: set[str] = set()
        gt_display = " | ".join(e for e in expected if not (e in seen or seen.add(e)))  # type: ignore[func-returns-value]
    else:
        gt_display = expected
    try:
        result = llm.chat([{
            "role": "user",
            "content": (
                f"You are evaluating clinical information extraction quality.\n\n"
                f"Field: {field}\n"
                f"Acceptable ground truth values: \"{gt_display}\"\n"
                f"Model extracted: \"{predicted}\"\n\n"
                f"Source text:\n{source_text[:3000]}\n\n"
                "Is the extracted value semantically correct for this field?\n"
                "Rules:\n"
                "- CORRECT if it matches or is semantically equivalent to ANY of the acceptable ground truth values\n"
                "- CORRECT if it refers to the same clinical concept, even if abbreviated or phrased differently\n"
                "- CORRECT if it is a more complete / more specific version of any ground truth value\n"
                "- CORRECT if any ground truth value is an abbreviation/acronym of the extracted value\n"
                "- WRONG if it refers to an entirely different concept from all ground truth values\n"
                "- WRONG if it is a result or finding rather than the measure/intervention name\n"
                "- WRONG if it is empty\n\n"
                'Return JSON: {"correct": true or false}'
            ),
        }], schema=_JUDGE_SCHEMA)
        return bool(result.get("correct", False))
    except Exception:
        return False


def _accuracy(
    extracted: dict[str, str],
    ground_truth: dict[str, str],
    skip: set[str],
    dataset: str = "clinicaltrials",
    all_gt_spans: dict[str, list[str]] | None = None,
) -> dict[str, bool]:
    """Score extracted fields against ground truth.

    When all_gt_spans is provided (EBM-NLP), a prediction is correct if it
    matches ANY annotated span for that field, not just the first one.
    """
    return {
        field: _score_field(
            field,
            extracted.get(field, ""),
            all_gt_spans.get(field, [ground_truth[field]]) if all_gt_spans else ground_truth[field],
            dataset,
        )
        for field in ground_truth
        if field not in skip
    }


def _apply_llm_judging(
    result: dict,
    sample: dict,
    skip: set[str],
    llm,
) -> None:
    """Re-score LLM_JUDGE_FIELDS in-place using semantic LLM judge.

    Only upgrades a score from False → True; never downgrades a passing field.
    Uses all_gt_spans when available so the judge sees every valid GT span.
    Updates field_acc to reflect the new scores.
    """
    acc = result.get("accuracy", {})
    extracted = result.get("extracted", {})
    all_gt_spans = sample.get("all_gt_spans", {})
    changed = False
    for field in _LLM_JUDGE_FIELDS:
        if field in skip or field not in acc:
            continue
        if acc[field]:
            continue  # already correct, skip LLM call
        expected = all_gt_spans.get(field) or sample["ground_truth"].get(field, "")
        judged = _llm_judge(field, extracted.get(field, ""), expected, sample["text"], llm)
        if judged:
            acc[field] = True
            changed = True
    if changed and acc:
        result["field_acc"] = sum(acc.values()) / len(acc)


# ---------------------------------------------------------------------------
# Prompt optimisation helpers
# ---------------------------------------------------------------------------

def _derive_lx_prompt(optimized_prompt: str, fields: list[str]) -> str:
    """Extract the instruction portion of a veritract-optimized prompt for LangExtract.

    Strips veritract-specific sections that conflict with LangExtract's QA format:
    - JSON structure examples ({...})
    - Source Text / Text: template placeholders
    - Examples: / Now extract sections
    Falls back to the default field-list description if nothing survives stripping.
    """
    lines = []
    for line in optimized_prompt.split("\n"):
        low = line.strip().lower().lstrip("*# ")  # strip markdown formatting before checking
        if (low.startswith("examples:") or low.startswith("text:")
                or low.startswith("source text") or low.startswith("now extract")
                or "{}" in line or (line.strip().startswith("{") and "}" in line)):
            break
        # Skip JSON-format instructions that conflict with LangExtract's QA format.
        if "return json" in low or "output json" in low:
            continue
        lines.append(line)
    instruction = "\n".join(lines).strip()
    if not instruction:
        return _lx_prompt_description(fields)
    return instruction


# ---------------------------------------------------------------------------
# veritract runner
# ---------------------------------------------------------------------------

def run_veritract_multi(
    samples: list[dict],
    model: str,
    schema: dict,
    dataset: str = "clinicaltrials",
    llm_judge: bool = False,
    optimized_prompt: str | None = None,
    n_runs: int = 1,
    base_seed: int = 42,
    temperature: float = 0.0,
) -> list[dict]:
    """Run veritract N times with full grounding; store spans and quarantine with raw_value.

    Returns a flat list of result rows across all (run, sample) pairs.
    Each row stores per-field spans (char offsets + provenance_type + confidence)
    and quarantined entries with the raw extracted value for downstream analysis.
    """
    fields = list(schema.get("properties", {}).keys())
    skip = UNSCORED_FIELDS.get(dataset, set())
    results: list[dict] = []

    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        llm = LLMClient(model=model, temperature=temperature, seed=seed)

        for s in samples:
            t0 = time.perf_counter()
            try:
                if optimized_prompt is not None:
                    full_prompt = optimized_prompt + f"\n\nText:\n{s['text'][:6000]}"
                    raw = vt_extract_raw(s["text"], schema, llm, prompt=full_prompt)
                else:
                    raw = vt_extract_raw(s["text"], schema, llm, examples=_SHARED_EXAMPLES)
                latency_extraction = time.perf_counter() - t0

                t1 = time.perf_counter()
                result = vt_ground(raw, llm, mode="full")
                latency_grounding = time.perf_counter() - t1

                extracted_vals: dict[str, str] = {}
                spans: dict[str, dict | None] = {}
                for field, gf in result.extracted.items():
                    extracted_vals[field] = gf["value"]
                    span = gf["span"]
                    spans[field] = {
                        "char_start": span["char_start"],
                        "char_end": span["char_end"],
                        "provenance_type": span["provenance_type"],
                        "confidence": gf["confidence"],
                    } if span is not None else None

                quarantined = [
                    {"field": q["field_name"], "raw_value": q["value"], "reason": q["reason"]}
                    for q in result.quarantined
                ]

                acc = _accuracy(extracted_vals, s["ground_truth"], skip, dataset, all_gt_spans=s.get("all_gt_spans"))
                n_with_span = sum(1 for f in fields if spans.get(f) is not None)
                row = {
                    "id": s["id"],
                    "run": run_idx,
                    "seed": seed,
                    "latency_extraction": latency_extraction,
                    "latency_grounding": latency_grounding,
                    "latency": latency_extraction + latency_grounding,
                    "extracted": extracted_vals,
                    "spans": spans,
                    "quarantined": quarantined,
                    "accuracy": acc,
                    "field_acc": sum(acc.values()) / len(acc) if acc else 0.0,
                    "grounding_rate": n_with_span / len(fields) if fields else 0.0,
                    "quarantine_rate": len(quarantined) / len(fields) if fields else 0.0,
                    "error": None,
                }
                if llm_judge:
                    _apply_llm_judging(row, s, skip, llm)
                results.append(row)

            except Exception as e:
                elapsed = time.perf_counter() - t0
                results.append({"id": s["id"], "run": run_idx, "seed": seed,
                                 "latency": elapsed, "error": str(e)})

    return results


# ---------------------------------------------------------------------------
# LangExtract runner
# ---------------------------------------------------------------------------

_LX_FIELD_DESCRIPTIONS = {
    "drug": "intervention drug name (verbatim from text, including dosage)",
    "sample_size": "number of participants enrolled (verbatim from text)",
    "outcome": "primary outcome measure name (not the result value)",
    "duration": "study length or follow-up period",
}


def _lx_examples(fields: list[str]):
    """Build LangExtract ExampleData list from shared few-shot examples."""
    try:
        from langextract.data import ExampleData, Extraction
    except ImportError:
        return []
    return [
        ExampleData(
            text=ex["text"],
            extractions=[
                Extraction(extraction_class=f, extraction_text=ex["fields"][f])
                for f in fields if f in ex["fields"]
            ],
        )
        for ex in _SHARED_EXAMPLES
    ]


def _lx_prompt_description(fields: list[str]) -> str:
    """Minimal prompt_description for LangExtract's QA template (field names + brief hint)."""
    field_list = ", ".join(
        f"{f} ({_LX_FIELD_DESCRIPTIONS.get(f, f)})" for f in fields
    )
    return (
        f"Extract the following fields from this clinical trial abstract, "
        f"copying the exact verbatim phrase from the text: {field_list}."
    )


def _lx_extract_once(
    sample: dict,
    model: str,
    fields: list[str],
    examples,
    seed: int,
    temperature: float,
    optimized_prompt: str | None,
) -> tuple[dict[str, str], bool, float]:
    """Run LangExtract on one sample. Returns (extracted_vals, failed, elapsed_s).

    If optimized_prompt is provided, its instruction portion (stripped of
    veritract-specific JSON template and Source Text sections) is passed as
    prompt_description so LangExtract's QA format is preserved.
    """
    import langextract as lx

    if optimized_prompt is not None:
        prompt_desc = _derive_lx_prompt(optimized_prompt, fields)
    else:
        prompt_desc = _lx_prompt_description(fields)

    t0 = time.perf_counter()
    docs = lx.extract(
        text_or_documents=sample["text"],
        model_id=model,
        prompt_description=prompt_desc,
        examples=examples,
        show_progress=False,
        temperature=temperature,
        max_char_buffer=8000,
        language_model_params={"seed": seed},
    )
    elapsed = time.perf_counter() - t0

    doc = docs if not isinstance(docs, list) else (docs[0] if docs else None)
    extracted: dict[str, str] = {}
    if doc and doc.extractions:
        for ext in doc.extractions:
            cls = (ext.extraction_class or "").lower()
            if cls in fields and cls not in extracted:
                extracted[cls] = ext.extraction_text or ""

    failed = not extracted
    return extracted, failed, elapsed


def run_langextract_multi(
    samples: list[dict],
    model: str,
    schema: dict,
    dataset: str = "clinicaltrials",
    llm_judge: bool = False,
    optimized_prompt: str | None = None,
    n_runs: int = 1,
    base_seed: int = 42,
    temperature: float = 0.0,
) -> list[dict]:
    """Run LangExtract N times, then apply veritract grounding; store spans and quarantine.

    Returns a flat list of result rows (one row per (run, sample) pair).
    Grounding is applied post-hoc via vt_ground(mode="full") so span metadata
    is captured in the same format as run_veritract_multi output.
    """
    try:
        import langextract as lx  # noqa: F401
    except ImportError:
        print("  [skip] langextract not installed — skipping LangExtract benchmark")
        return []

    fields = list(schema.get("properties", {}).keys())
    skip = UNSCORED_FIELDS.get(dataset, set())
    examples = _lx_examples(fields)
    results: list[dict] = []

    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        grounding_llm = LLMClient(model=model, temperature=temperature, seed=seed)

        for s in samples:
            try:
                extracted_vals, failed, latency_extraction = _lx_extract_once(
                    s, model, fields, examples, seed, temperature,
                    optimized_prompt=optimized_prompt,
                )
            except Exception as e:
                results.append({"id": s["id"], "run": run_idx, "seed": seed,
                                 "latency": 0.0, "error": str(e)})
                continue

            if failed:
                results.append({
                    "id": s["id"], "run": run_idx, "seed": seed,
                    "latency": latency_extraction,
                    "error": "extraction_failed: model returned no valid extractions",
                })
                continue

            lx_raw = RawExtractionResult(
                fields=extracted_vals,
                garbage=[],
                source_text=s["text"],
                doc_id=s["id"],
                source_type="text",
            )
            t1 = time.perf_counter()
            grounded = vt_ground(lx_raw, grounding_llm, mode="full")
            latency_grounding = time.perf_counter() - t1

            grounded_vals: dict[str, str] = {}
            spans: dict[str, dict | None] = {}
            for field, gf in grounded.extracted.items():
                grounded_vals[field] = gf["value"]
                span = gf["span"]
                spans[field] = {
                    "char_start": span["char_start"],
                    "char_end": span["char_end"],
                    "provenance_type": span["provenance_type"],
                    "confidence": gf["confidence"],
                } if span is not None else None

            quarantined = [
                {"field": q["field_name"], "raw_value": q["value"], "reason": q["reason"]}
                for q in grounded.quarantined
            ]

            acc = _accuracy(grounded_vals, s["ground_truth"], skip, dataset, all_gt_spans=s.get("all_gt_spans"))
            n_with_span = sum(1 for f in fields if spans.get(f) is not None)
            row = {
                "id": s["id"],
                "run": run_idx,
                "seed": seed,
                "latency_extraction": latency_extraction,
                "latency_grounding": latency_grounding,
                "latency": latency_extraction + latency_grounding,
                "extracted": grounded_vals,
                "spans": spans,
                "quarantined": quarantined,
                "accuracy": acc,
                "field_acc": sum(acc.values()) / len(acc) if acc else 0.0,
                "grounding_rate": n_with_span / len(fields) if fields else 0.0,
                "quarantine_rate": len(quarantined) / len(fields) if fields else 0.0,
                "error": None,
            }
            if llm_judge:
                _apply_llm_judging(row, s, skip, grounding_llm)
            results.append(row)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def print_summary(name: str, results: list[dict], fields: list[str] | None = None, ci_half: float = 0.0) -> dict:
    ok = [r for r in results if not r.get("error")]
    if not ok:
        print(f"\n{name}: no successful results")
        return {}

    latencies = [r["latency"] for r in ok]
    field_accs = [r["field_acc"] for r in ok]
    grounding_rates = [r["grounding_rate"] for r in ok]

    all_fields = fields or sorted({f for r in ok for f in r.get("accuracy", {})})
    per_field_acc: dict[str, list[bool]] = {f: [] for f in all_fields}
    for r in ok:
        for f, correct in r.get("accuracy", {}).items():
            per_field_acc.setdefault(f, []).append(correct)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Samples:           {len(ok)}/{len(results)}")
    print(f"  Latency (mean):    {_mean(latencies):.1f}s")
    print(f"  Latency (min/max): {min(latencies):.1f}s / {max(latencies):.1f}s")
    ci_str = f" ± {ci_half*100:.1f}%" if ci_half > 0 else ""
    print(f"  Field accuracy:    {_mean(field_accs)*100:.1f}%{ci_str}")
    print(f"  Grounding rate:    {_mean(grounding_rates)*100:.1f}%")
    if any(r.get("quarantine_rate") is not None for r in ok):
        qr = [r["quarantine_rate"] for r in ok if r.get("quarantine_rate") is not None]
        print(f"  Quarantine rate:   {_mean(qr)*100:.1f}%")
    scored = [f for f in all_fields if f not in UNSCORED_FIELDS]
    unscored = [f for f in all_fields if f in UNSCORED_FIELDS]
    print(f"\n  Per-field accuracy (scored):")
    for f in scored:
        vals = per_field_acc.get(f, [])
        if vals:
            print(f"    {f:<20} {sum(vals)/len(vals)*100:.0f}%")
    if unscored:
        print(f"  Unscored fields (shown for reference):")
        for f in unscored:
            vals = per_field_acc.get(f, [])
            if vals:
                print(f"    {f:<20} {sum(vals)/len(vals)*100:.0f}%  (not in accuracy metric)")

    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for r in errors:
            print(f"    {r['id']}: {r['error'][:80]}")

    return {
        "n": len(ok),
        "latency_mean": _mean(latencies),
        "field_accuracy": _mean(field_accs),
        "grounding_rate": _mean(grounding_rates),
    }


def print_grounding_metrics(name: str, results: list[dict], fields: list[str]) -> None:
    """Print provenance breakdown and quarantine precision/recall."""
    ok = [r for r in results if not r.get("error")]
    if not ok:
        return

    provenance_counts: dict[str, int] = {}
    total_grounded = 0
    for r in ok:
        for span_info in r.get("spans", {}).values():
            if span_info is not None:
                ptype = span_info.get("provenance_type", "unknown")
                provenance_counts[ptype] = provenance_counts.get(ptype, 0) + 1
                total_grounded += 1

    n_quarantine_correct = 0
    n_quarantine_wrong = 0
    n_not_quarantined_wrong = 0
    n_total_quarantined = 0
    for r in ok:
        acc = r.get("accuracy", {})
        quarantined_fields = {q["field"] for q in r.get("quarantined", [])}
        n_total_quarantined += len(quarantined_fields)
        for field, is_correct in acc.items():
            if field in quarantined_fields:
                if is_correct:
                    n_quarantine_correct += 1
                else:
                    n_quarantine_wrong += 1
            elif not is_correct:
                n_not_quarantined_wrong += 1

    precision = n_quarantine_wrong / n_total_quarantined if n_total_quarantined else 0.0
    denom_recall = n_quarantine_wrong + n_not_quarantined_wrong
    recall = n_quarantine_wrong / denom_recall if denom_recall else 0.0

    print(f"\n  {name} — Grounding metrics:")
    if total_grounded:
        print(f"  Provenance ({total_grounded} grounded fields):")
        for ptype, count in sorted(provenance_counts.items(), key=lambda x: -x[1]):
            print(f"    {ptype:<15} {count:>5}  ({count/total_grounded*100:.0f}%)")
    if n_total_quarantined:
        print(f"  Quarantine precision: {precision*100:.0f}%"
              f"  ({n_quarantine_wrong}/{n_total_quarantined} quarantined were actually wrong)")
        print(f"  Quarantine recall:    {recall*100:.0f}%"
              f"  ({n_quarantine_wrong}/{denom_recall} wrong fields caught)")
    else:
        print(f"  No quarantined fields.")


def print_comparison(
    summaries: dict[str, dict],  # label → summary dict
) -> None:
    active = {k: v for k, v in summaries.items() if v}
    if len(active) < 2:
        return
    labels = list(active.keys())
    col_w = 13
    print(f"\n{'='*50}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*50}")
    header = f"  {'Metric':<22}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print(f"  {'-'*(22 + col_w * len(labels))}")
    for key, label in [
        ("latency_mean", "Latency (s)"),
        ("field_accuracy", "Field accuracy"),
        ("grounding_rate", "Grounding rate"),
    ]:
        row = f"  {label:<22}"
        for lbl in labels:
            v = active[lbl].get(key, 0)
            row += f"{v:>11.1f}s" if key == "latency_mean" else f"{v*100:>11.1f}%"
            row += " " * (col_w - 12)
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="veritract vs LangExtract benchmark")
    parser.add_argument("--model", default="gemma4:e4b", help="Ollama model to use")
    parser.add_argument("--no-veritract", action="store_true", help="Skip veritract")
    parser.add_argument("--no-langextract", action="store_true", help="Skip LangExtract")
    parser.add_argument("--llm-judge", action="store_true",
                        help="Re-score outcome/drug fields with LLM semantic judge (adds latency)")
    parser.add_argument("--samples", type=int, default=0, help="Limit to N samples (0=all)")
    parser.add_argument("--out", default="", help="Save JSON results to this file")
    parser.add_argument("--vt-results", default="", help="Load veritract results from a prior JSON run")
    parser.add_argument("--optimize", action="store_true",
                        help="Run prompt optimization before benchmark using a calibration subset")
    parser.add_argument("--opt-samples", type=int, default=20,
                        help="Number of samples to use for optimization calibration (default: 20)")
    parser.add_argument("--opt-n", type=int, default=3,
                        help="Number of optimization iterations (default: 3)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per arm with different seeds for CI (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM sampling temperature; 0.0 = deterministic (default: 0.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed; run i uses seed+i (default: 42)")
    parser.add_argument(
        "--dataset", choices=["synthetic", "clinicaltrials", "ebmnlp"], default="ebmnlp",
        help="Dataset to use (default: ebmnlp — requires --build first)",
    )
    args = parser.parse_args()

    if args.dataset == "clinicaltrials":
        try:
            all_samples = get_ct_samples()
            schema = CT_SCHEMA
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif args.dataset == "ebmnlp":
        try:
            all_samples = get_ebmnlp_samples()
            schema = EBMNLP_SCHEMA
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        all_samples = SYNTHETIC_SAMPLES
        schema = SYNTHETIC_SCHEMA

    samples = all_samples[:args.samples] if args.samples else all_samples

    judge = args.llm_judge
    print(f"Benchmark: veritract vs LangExtract")
    print(f"Model: {args.model} | Dataset: {args.dataset} | Samples: {len(samples)}"
          + (f" | Runs: {args.runs}" if args.runs > 1 else "")
          + (f" | Temp: {args.temperature}" if hasattr(args, 'temperature') else "")
          + (" | LLM judge: ON" if judge else ""))
    print(f"{'='*50}")

    optimized_prompt: str | None = None
    if args.optimize:
        n_cal = min(args.opt_samples, len(all_samples))
        cal_samples = all_samples[:n_cal]
        cal_examples = [{"text": s["text"], "fields": s["ground_truth"]} for s in cal_samples]
        cal_gt = [s["ground_truth"] for s in cal_samples]
        opt_llm = LLMClient(model=args.model)
        print(f"\nRunning prompt optimization ({args.opt_n} iterations on {n_cal} calibration samples)...")
        optimized_prompt = vt_optimize_prompt(
            cal_examples, schema, opt_llm,
            n_iter=args.opt_n, ground_truth=cal_gt,
        )
        print(f"\nOptimized prompt (first 600 chars):\n{optimized_prompt[:600]}")
        print(f"{'='*50}")

    fields = list(schema.get("properties", {}).keys())
    summaries: dict[str, dict] = {}
    all_results: dict[str, list[dict]] = {}
    n_runs = args.runs

    # ------------------------------------------------------------------
    # veritract: extract_raw + full grounding; spans stored per field
    # ------------------------------------------------------------------
    if not args.no_veritract:
        print(f"\nRunning veritract ({n_runs} run(s), temp={args.temperature}, base_seed={args.seed})...")
        vt_results = run_veritract_multi(
            samples, args.model, schema,
            dataset=args.dataset,
            llm_judge=judge,
            optimized_prompt=optimized_prompt,
            n_runs=n_runs,
            base_seed=args.seed,
            temperature=args.temperature,
        )
        run_means = [
            _mean([r["field_acc"] for r in vt_results if r.get("run") == ri and not r.get("error")])
            for ri in range(n_runs)
        ]
        _, ci_half = _ci95(run_means)
        summaries["veritract"] = print_summary("veritract", vt_results, fields, ci_half=ci_half)
        print_grounding_metrics("veritract", vt_results, fields)
        all_results["veritract"] = vt_results

    # ------------------------------------------------------------------
    # LangExtract: extract N times, then apply veritract grounding post-hoc
    # ------------------------------------------------------------------
    if not args.no_langextract:
        print(f"\nRunning LangExtract ({n_runs} run(s), temp={args.temperature}, base_seed={args.seed})...")
        lx_results = run_langextract_multi(
            samples, args.model, schema,
            dataset=args.dataset,
            llm_judge=judge,
            optimized_prompt=optimized_prompt,
            n_runs=n_runs,
            base_seed=args.seed,
            temperature=args.temperature,
        )
        run_means = [
            _mean([r["field_acc"] for r in lx_results if r.get("run") == ri and not r.get("error")])
            for ri in range(n_runs)
        ]
        _, ci_half = _ci95(run_means)
        summaries["langextract"] = print_summary("LangExtract", lx_results, fields, ci_half=ci_half)
        print_grounding_metrics("LangExtract", lx_results, fields)
        all_results["langextract"] = lx_results

    print_comparison(summaries)

    if args.out:
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
