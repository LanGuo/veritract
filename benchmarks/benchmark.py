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


def _score_field(field: str, predicted: str, expected: str, dataset: str = "clinicaltrials") -> bool:
    """Field-specific scoring logic."""
    if field == "sample_size":
        # Try numeric proximity first (handles CT.gov enrolled vs analyzed N,
        # and EBM-NLP written-out numbers when LLM normalizes to digits).
        gt_n = _extract_number(expected)
        ex_n = _extract_number(predicted)
        if gt_n and ex_n and abs(gt_n - ex_n) / gt_n <= 0.15:
            return True
        # Fall back to fuzzy (handles written-out numbers like "Two hundred...")
    return _fuzzy_match(predicted, expected)


def _llm_judge(field: str, predicted: str, expected: str, source_text: str, llm) -> bool:
    """Semantic equivalence judge via LLM with a clinical extraction rubric.

    Used for fields (outcome, drug) where fuzzy string matching is unreliable
    because valid extractions may be paraphrased, abbreviated, or more/less
    specific than the ground truth span.
    """
    if not predicted.strip():
        return False
    try:
        result = llm.chat([{
            "role": "user",
            "content": (
                f"You are evaluating clinical information extraction quality.\n\n"
                f"Field: {field}\n"
                f"Ground truth: \"{expected}\"\n"
                f"Model extracted: \"{predicted}\"\n\n"
                f"Source text:\n{source_text[:3000]}\n\n"
                "Is the extracted value semantically correct for this field?\n"
                "Rules:\n"
                "- CORRECT if it refers to the same clinical concept, even if abbreviated or phrased differently\n"
                "- CORRECT if it is a more complete / more specific version of the ground truth\n"
                "- CORRECT if the ground truth is an abbreviation/acronym of the extracted value\n"
                "- WRONG if it refers to an entirely different concept\n"
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
) -> dict[str, bool]:
    return {
        field: _score_field(field, extracted.get(field, ""), ground_truth[field], dataset)
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
    Updates field_acc to reflect the new scores.
    """
    acc = result.get("accuracy", {})
    extracted = result.get("extracted", {})
    changed = False
    for field in _LLM_JUDGE_FIELDS:
        if field in skip or field not in acc:
            continue
        if acc[field]:
            continue  # already correct, skip LLM call
        judged = _llm_judge(
            field,
            extracted.get(field, ""),
            sample["ground_truth"].get(field, ""),
            sample["text"],
            llm,
        )
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
    modes: tuple[str, ...] = ("full", "no-grounding"),
    llm_judge: bool = False,
    optimized_prompt: str | None = None,
    n_runs: int = 1,
    base_seed: int = 42,
    temperature: float = 0.0,
) -> dict[str, list[dict]]:
    """Run veritract N times; apply all grounding modes to the same raw output per run.

    Each run uses a different seed (base_seed + run_idx) with temperature=0 for
    near-deterministic extraction. All modes share the same extract_raw() output
    within a run, enabling a true apples-to-apples grounding comparison.

    Returns dict mapping mode name → flat list of result rows across all runs.
    Each row has a "run" key with the run index.
    """
    fields = list(schema.get("properties", {}).keys())
    skip = UNSCORED_FIELDS.get(dataset, set())
    all_mode_results: dict[str, list[dict]] = {m: [] for m in modes}

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
                raw_elapsed = time.perf_counter() - t0

                for mode in modes:
                    t1 = time.perf_counter()
                    result = vt_ground(raw, llm, mode=mode)
                    elapsed = raw_elapsed + (time.perf_counter() - t1)

                    extracted_vals = {k: v["value"] for k, v in result.extracted.items()}
                    acc = _accuracy(extracted_vals, s["ground_truth"], skip, dataset)
                    grounded = sum(1 for v in extracted_vals.values() if _is_verbatim(v, s["text"]))
                    total_extracted = len(extracted_vals)
                    row = {
                        "id": s["id"],
                        "run": run_idx,
                        "seed": seed,
                        "latency": elapsed,
                        "accuracy": acc,
                        "field_acc": sum(acc.values()) / len(acc) if acc else 0.0,
                        "grounding_rate": grounded / total_extracted if total_extracted else 0.0,
                        "quarantine_rate": len(result.quarantined) / len(fields),
                        "extracted": extracted_vals,
                        "quarantined": [q["field_name"] for q in result.quarantined],
                        "error": None,
                    }
                    if llm_judge:
                        _apply_llm_judging(row, s, skip, llm)
                    all_mode_results[mode].append(row)

            except Exception as e:
                elapsed = time.perf_counter() - t0
                err = {"id": s["id"], "run": run_idx, "latency": elapsed, "error": str(e)}
                for mode in modes:
                    all_mode_results[mode].append(err)

    return all_mode_results


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


def _score_extracted(
    extracted_vals: dict[str, str],
    sample: dict,
    skip: set[str],
    dataset: str,
    run_idx: int,
    seed: int,
    latency: float,
    quarantine_rate: float | None = None,
    quarantined: list[str] | None = None,
    llm=None,
) -> dict:
    acc = _accuracy(extracted_vals, sample["ground_truth"], skip, dataset)
    grounded_count = sum(1 for v in extracted_vals.values() if _is_verbatim(v, sample["text"]))
    total = len(extracted_vals) or 1
    row = {
        "id": sample["id"],
        "run": run_idx,
        "seed": seed,
        "latency": latency,
        "accuracy": acc,
        "field_acc": sum(acc.values()) / len(acc) if acc else 0.0,
        "grounding_rate": grounded_count / total,
        "quarantine_rate": quarantine_rate,
        "extracted": extracted_vals,
        "quarantined": quarantined or [],
        "error": None,
    }
    if llm:
        _apply_llm_judging(row, sample, skip, llm)
    return row


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
) -> dict[str, list[dict]]:
    """Run LangExtract N times (different seeds), then apply veritract grounding post-hoc.

    Returns {"raw": [...], "grounded": [...]} — parallel lists, one row per (run, sample).
    The "raw" arm scores LangExtract output verbatim; "grounded" passes the same values
    through vt_ground(mode="full") for two-stage verification and quarantine.
    """
    try:
        import langextract as lx  # noqa: F401
    except ImportError:
        print("  [skip] langextract not installed — skipping LangExtract benchmark")
        return {"raw": [], "grounded": []}

    fields = list(schema.get("properties", {}).keys())
    skip = UNSCORED_FIELDS.get(dataset, set())
    examples = _lx_examples(fields)
    all_raw: list[dict] = []
    all_grounded: list[dict] = []

    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        grounding_llm = LLMClient(model=model, temperature=temperature, seed=seed)

        for s in samples:
            try:
                extracted_vals, failed, elapsed = _lx_extract_once(
                    s, model, fields, examples, seed, temperature,
                    optimized_prompt=optimized_prompt,
                )
            except Exception as e:
                err = {"id": s["id"], "run": run_idx, "latency": 0.0, "error": str(e)}
                all_raw.append(err)
                all_grounded.append(err)
                continue

            if failed:
                err = {"id": s["id"], "run": run_idx, "seed": seed, "latency": elapsed,
                       "error": "extraction_failed: model returned no valid extractions"}
                all_raw.append(err)
                all_grounded.append(err)
                continue

            judge_llm = grounding_llm if llm_judge else None

            # Raw arm: LangExtract output without grounding
            all_raw.append(_score_extracted(
                extracted_vals, s, skip, dataset, run_idx, seed, elapsed, llm=judge_llm,
            ))

            # Grounded arm: wrap as RawExtractionResult → vt_ground(mode="full")
            lx_raw = RawExtractionResult(
                fields=extracted_vals,
                garbage=[],
                source_text=s["text"],
                doc_id=s["id"],
                source_type="text",
            )
            t1 = time.perf_counter()
            grounded = vt_ground(lx_raw, grounding_llm, mode="full")
            ground_elapsed = elapsed + (time.perf_counter() - t1)

            grounded_vals = {k: v["value"] for k, v in grounded.extracted.items()}
            all_grounded.append(_score_extracted(
                grounded_vals, s, skip, dataset, run_idx, seed, ground_elapsed,
                quarantine_rate=len(grounded.quarantined) / len(fields),
                quarantined=[q["field_name"] for q in grounded.quarantined],
                llm=judge_llm,
            ))

    return {"raw": all_raw, "grounded": all_grounded}


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
    parser.add_argument("--no-reground", action="store_true", help="Disable veritract auto_reground")
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
    # veritract: extract_raw once per (run, sample), ground post-hoc
    #   "no-grounding" arm = raw extractor output (pre-grounding)
    #   "full" arm         = same raw + two-stage grounding
    # ------------------------------------------------------------------
    if not args.no_veritract:
        print(f"\nRunning veritract ({n_runs} run(s), temp={args.temperature}, base_seed={args.seed})...")
        mode_results = run_veritract_multi(
            samples, args.model, schema,
            dataset=args.dataset,
            modes=("full", "no-grounding"),
            llm_judge=judge,
            optimized_prompt=optimized_prompt,
            n_runs=n_runs,
            base_seed=args.seed,
            temperature=args.temperature,
        )
        arm_map = {"no-grounding": "veritract_raw", "full": "veritract_grounded"}
        label_map = {"no-grounding": "veritract (raw)", "full": "veritract (grounded)"}
        for mode_name, results in mode_results.items():
            label = label_map[mode_name]
            run_means = [
                _mean([r["field_acc"] for r in results if r.get("run") == ri and not r.get("error")])
                for ri in range(n_runs)
            ]
            _, ci_half = _ci95(run_means)
            summaries[label] = print_summary(label, results, fields, ci_half=ci_half)
            all_results[arm_map[mode_name]] = results

    # ------------------------------------------------------------------
    # LangExtract: extract N times (different seeds), then ground post-hoc
    #   "raw" arm     = LangExtract output verbatim (no grounding)
    #   "grounded" arm = same output + veritract two-stage grounding
    # ------------------------------------------------------------------
    if not args.no_langextract:
        print(f"\nRunning LangExtract ({n_runs} run(s), temp={args.temperature}, base_seed={args.seed})...")
        lx_multi = run_langextract_multi(
            samples, args.model, schema,
            dataset=args.dataset,
            llm_judge=judge,
            optimized_prompt=optimized_prompt,
            n_runs=n_runs,
            base_seed=args.seed,
            temperature=args.temperature,
        )
        lx_arm_map = {"raw": "langextract_raw", "grounded": "langextract_grounded"}
        lx_label_map = {"raw": "LangExtract (raw)", "grounded": "LangExtract (grounded)"}
        for arm, results in lx_multi.items():
            label = lx_label_map[arm]
            run_means = [
                _mean([r["field_acc"] for r in results if r.get("run") == ri and not r.get("error")])
                for ri in range(n_runs)
            ]
            _, ci_half = _ci95(run_means)
            summaries[label] = print_summary(label, results, fields, ci_half=ci_half)
            all_results[lx_arm_map[arm]] = results

    print_comparison(summaries)

    if args.out:
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
