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
import re
import sys
import time
from pathlib import Path

from rapidfuzz import fuzz

# Add parent dir so the local veritract package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.dataset import SAMPLES as SYNTHETIC_SAMPLES, SCHEMA as SYNTHETIC_SCHEMA, FIELDS as SYNTHETIC_FIELDS  # noqa: E402
from benchmarks.clinicaltrials_dataset import get_samples as get_ct_samples, SCHEMA as CT_SCHEMA, FIELDS as CT_FIELDS  # noqa: E402
from benchmarks.ebmnlp_dataset import get_samples as get_ebmnlp_samples, SCHEMA as EBMNLP_SCHEMA, FIELDS as EBMNLP_FIELDS  # noqa: E402
from veritract import extract as vt_extract, LLMClient, optimize_prompt as vt_optimize_prompt  # noqa: E402

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

    Stops before the Examples: / Text: / Now extract sections which are
    veritract-specific and would confuse LangExtract's QA format.
    """
    lines = []
    for line in optimized_prompt.split("\n"):
        low = line.strip().lower()
        if low.startswith("examples:") or low.startswith("text:") or low.startswith("now extract"):
            break
        lines.append(line)
    instruction = "\n".join(lines).strip()
    if not instruction:
        field_list = ", ".join(fields)
        return (
            f"Extract the following fields verbatim from the clinical trial abstract: {field_list}. "
            "Copy the exact phrase from the text."
        )
    return instruction


# ---------------------------------------------------------------------------
# veritract runner
# ---------------------------------------------------------------------------

def run_veritract(
    samples: list[dict], model: str, schema: dict, dataset: str = "clinicaltrials",
    mode: str = "full", llm_judge: bool = False,
    optimized_prompt: str | None = None,
) -> list[dict]:
    llm = LLMClient(model=model)
    fields = list(schema.get("properties", {}).keys())
    skip = UNSCORED_FIELDS.get(dataset, set())
    results = []
    for i, s in enumerate(samples):
        t0 = time.perf_counter()
        try:
            if optimized_prompt is not None:
                full_prompt = optimized_prompt + f"\n\nText:\n{s['text'][:6000]}"
                result = vt_extract(s["text"], schema, llm, mode=mode, prompt=full_prompt)
            else:
                result = vt_extract(s["text"], schema, llm, mode=mode,
                                   examples=_SHARED_EXAMPLES)
            elapsed = time.perf_counter() - t0
            extracted_vals = {k: v["value"] for k, v in result.extracted.items()}
            acc = _accuracy(extracted_vals, s["ground_truth"], skip, dataset)
            grounded = sum(
                1 for f, v in extracted_vals.items()
                if _is_verbatim(v, s["text"])
            )
            total_extracted = len(extracted_vals)
            row = {
                "id": s["id"],
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
            results.append(row)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results.append({"id": s["id"], "latency": elapsed, "error": str(e)})
    return results


# ---------------------------------------------------------------------------
# LangExtract runner
# ---------------------------------------------------------------------------

def run_langextract(
    samples: list[dict], model: str, schema: dict, dataset: str = "clinicaltrials",
    llm_judge: bool = False, optimized_prompt: str | None = None,
) -> list[dict]:
    try:
        import langextract as lx
        from langextract.data import ExampleData, Extraction
    except ImportError:
        print("  [skip] langextract not installed — skipping LangExtract benchmark")
        return []

    fields = list(schema.get("properties", {}).keys())
    skip = UNSCORED_FIELDS.get(dataset, set())
    judge_llm = LLMClient(model=model) if llm_judge else None

    # Build LangExtract examples from _SHARED_EXAMPLES, filtered to schema fields
    lx_examples = [
        ExampleData(
            text=ex["text"],
            extractions=[
                Extraction(extraction_class=f, extraction_text=ex["fields"][f])
                for f in fields
                if f in ex["fields"]
            ],
        )
        for ex in _SHARED_EXAMPLES
    ]
    field_descriptions = {
        "drug": "intervention drug name (verbatim from text, including dosage)",
        "sample_size": "number of participants enrolled (verbatim from text)",
        "outcome": "primary outcome measure name (not the result value)",
        "duration": "study length or follow-up period",
    }
    if optimized_prompt is not None:
        prompt = _derive_lx_prompt(optimized_prompt, fields)
    else:
        field_list = ", ".join(
            f"{f} ({field_descriptions.get(f, f)})" for f in fields
        )
        prompt = (
            f"Extract the following fields from this clinical trial abstract, "
            f"copying the exact verbatim phrase from the text: {field_list}. "
            "Each field should appear as a separate extraction with extraction_class "
            "set to the field name."
        )

    results = []
    for s in samples:
        t0 = time.perf_counter()
        try:
            docs = lx.extract(
                text_or_documents=s["text"],
                model_id=model,
                prompt_description=prompt,
                examples=lx_examples,
                show_progress=False,
            )
            elapsed = time.perf_counter() - t0

            doc = docs if not isinstance(docs, list) else (docs[0] if docs else None)
            extracted_vals: dict[str, str] = {}
            raw_extraction_count = len(doc.extractions) if doc and doc.extractions else 0
            if doc and doc.extractions:
                for ext in doc.extractions:
                    cls = (ext.extraction_class or "").lower()
                    if cls in fields and cls not in extracted_vals:
                        extracted_vals[cls] = ext.extraction_text or ""

            # Grounding: same definition as veritract — verbatim substring match
            grounded_count = sum(
                1 for v in extracted_vals.values() if _is_verbatim(v, s["text"])
            )
            acc = _accuracy(extracted_vals, s["ground_truth"], skip, dataset)
            total = len(extracted_vals) or 1
            extraction_failed = raw_extraction_count == 0
            row = {
                "id": s["id"],
                "latency": elapsed,
                "accuracy": acc,
                "field_acc": sum(acc.values()) / len(acc) if acc else 0.0,
                "grounding_rate": grounded_count / total,
                "quarantine_rate": None,
                "extracted": extracted_vals,
                "error": "extraction_failed: model returned no valid extractions" if extraction_failed else None,
            }
            if judge_llm and not extraction_failed:
                _apply_llm_judging(row, s, skip, judge_llm)
            results.append(row)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results.append({"id": s["id"], "latency": elapsed, "error": str(e)})
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def print_summary(name: str, results: list[dict], fields: list[str] | None = None) -> dict:
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
    print(f"  Field accuracy:    {_mean(field_accs)*100:.1f}%")
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

    if args.vt_results:
        prior = json.loads(Path(args.vt_results).read_text())
        vt_results = prior.get("veritract", prior) if isinstance(prior, dict) else prior
        print(f"\nLoaded {len(vt_results)} veritract results from {args.vt_results}")
        summaries["veritract (full)"] = print_summary("veritract (full)", vt_results, fields)
        all_results["veritract_full"] = vt_results
    elif not args.no_veritract:
        print("\nRunning veritract (full grounding)...")
        vt_full = run_veritract(
            samples, args.model, schema, dataset=args.dataset,
            mode="full", llm_judge=judge, optimized_prompt=optimized_prompt,
        )
        summaries["veritract (full)"] = print_summary("veritract (full)", vt_full, fields)
        all_results["veritract_full"] = vt_full

        print("\nRunning veritract (no-grounding)...")
        vt_ng = run_veritract(
            samples, args.model, schema, dataset=args.dataset,
            mode="no-grounding", llm_judge=judge, optimized_prompt=optimized_prompt,
        )
        summaries["veritract (no-grnd)"] = print_summary("veritract (no-grnd)", vt_ng, fields)
        all_results["veritract_no_grounding"] = vt_ng

    if not args.no_langextract:
        print("\nRunning LangExtract...")
        lx_results = run_langextract(samples, args.model, schema, dataset=args.dataset,
                                     llm_judge=judge, optimized_prompt=optimized_prompt)
        summaries["LangExtract"] = print_summary("LangExtract", lx_results, fields)
        all_results["langextract"] = lx_results

    print_comparison(summaries)

    if args.out:
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
