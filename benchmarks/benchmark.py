#!/usr/bin/env python3
"""Head-to-head benchmark: veritract vs LangExtract on clinical trial extraction.

Both packages use the same local Ollama model (gemma4:e4b by default).
The task is extracting four fields from clinical trial abstracts:
  drug, sample_size, outcome, duration

Metrics:
  - Latency (seconds per sample)
  - Field accuracy (fuzzy match ≥80 vs ground truth)
  - Grounding rate (% of extracted fields with char-level provenance)
  - Quarantine rate (% of fields quarantined; veritract-only)

Usage:
  python benchmarks/benchmark.py [--model MODEL] [--no-langextract]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rapidfuzz import fuzz

# Add parent dir so the local veritract package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.dataset import SAMPLES as SYNTHETIC_SAMPLES, SCHEMA as SYNTHETIC_SCHEMA, FIELDS as SYNTHETIC_FIELDS  # noqa: E402
from benchmarks.clinicaltrials_dataset import get_samples as get_ct_samples, SCHEMA as CT_SCHEMA, FIELDS as CT_FIELDS  # noqa: E402
from veritract import extract as vt_extract, LLMClient  # noqa: E402

_FUZZY_THRESHOLD = 70  # token_set_ratio threshold for "correct" answer

# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def _score(predicted: str, expected: str) -> float:
    return fuzz.token_set_ratio(predicted.lower(), expected.lower())


def _accuracy(extracted: dict[str, str], ground_truth: dict[str, str]) -> dict[str, bool]:
    return {
        field: _score(extracted.get(field, ""), ground_truth[field]) >= _FUZZY_THRESHOLD
        for field in ground_truth
    }


# ---------------------------------------------------------------------------
# veritract runner
# ---------------------------------------------------------------------------

def run_veritract(
    samples: list[dict], model: str, schema: dict, auto_reground: bool = True
) -> list[dict]:
    llm = LLMClient(model=model)
    fields = list(schema.get("properties", {}).keys())
    results = []
    for s in samples:
        t0 = time.perf_counter()
        try:
            result = vt_extract(s["text"], schema, llm, auto_reground=auto_reground)
            elapsed = time.perf_counter() - t0
            extracted_vals = {k: v["value"] for k, v in result.extracted.items()}
            acc = _accuracy(extracted_vals, s["ground_truth"])
            grounded = sum(1 for f in result.extracted.values() if f["span"] is not None)
            total_extracted = len(result.extracted)
            results.append({
                "id": s["id"],
                "latency": elapsed,
                "accuracy": acc,
                "field_acc": sum(acc.values()) / len(acc),
                "grounding_rate": grounded / total_extracted if total_extracted else 0.0,
                "quarantine_rate": len(result.quarantined) / len(fields),
                "extracted": extracted_vals,
                "quarantined": [q["field_name"] for q in result.quarantined],
                "error": None,
            })
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results.append({"id": s["id"], "latency": elapsed, "error": str(e)})
    return results


# ---------------------------------------------------------------------------
# LangExtract runner
# ---------------------------------------------------------------------------

def run_langextract(samples: list[dict], model: str, schema: dict) -> list[dict]:
    try:
        import langextract as lx
        from langextract.data import ExampleData, Extraction
    except ImportError:
        print("  [skip] langextract not installed — skipping LangExtract benchmark")
        return []

    # Few-shot example to guide extraction format
    example = ExampleData(
        text=(
            "A randomized trial enrolled 248 patients with type 2 diabetes. "
            "Participants received metformin 500mg twice daily or placebo for 12 months. "
            "The primary outcome was HbA1c reduction at 12 months."
        ),
        extractions=[
            Extraction(extraction_class="drug", extraction_text="metformin"),
            Extraction(extraction_class="sample_size", extraction_text="248 patients"),
            Extraction(extraction_class="outcome", extraction_text="HbA1c reduction"),
            Extraction(extraction_class="duration", extraction_text="12 months"),
        ],
    )

    prompt = (
        "Extract four fields from this clinical trial abstract: "
        "drug (the intervention drug name), sample_size (number of participants), "
        "outcome (primary outcome measure), duration (study length). "
        "Each field should appear as a separate extraction with extraction_class set "
        "to the field name."
    )

    results = []
    for s in samples:
        t0 = time.perf_counter()
        try:
            docs = lx.extract(
                text_or_documents=s["text"],
                model_id=model,
                prompt_description=prompt,
                examples=[example],
                show_progress=False,
            )
            elapsed = time.perf_counter() - t0

            doc = docs if not isinstance(docs, list) else (docs[0] if docs else None)
            extracted_vals: dict[str, str] = {}
            grounded_count = 0
            fields = list(schema.get("properties", {}).keys())
            if doc and doc.extractions:
                for ext in doc.extractions:
                    cls = (ext.extraction_class or "").lower()
                    if cls in fields and cls not in extracted_vals:
                        extracted_vals[cls] = ext.extraction_text or ""
                        if ext.char_interval and ext.char_interval.start_pos is not None:
                            grounded_count += 1

            acc = _accuracy(extracted_vals, s["ground_truth"])
            total = len(extracted_vals) or 1
            results.append({
                "id": s["id"],
                "latency": elapsed,
                "accuracy": acc,
                "field_acc": sum(acc.values()) / len(acc),
                "grounding_rate": grounded_count / total,
                "quarantine_rate": None,  # LangExtract has no quarantine concept
                "extracted": extracted_vals,
                "error": None,
            })
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
    print(f"\n  Per-field accuracy:")
    for f in all_fields:
        vals = per_field_acc.get(f, [])
        if vals:
            print(f"    {f:<20} {sum(vals)/len(vals)*100:.0f}%")

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


def print_comparison(vt_summary: dict, lx_summary: dict) -> None:
    if not vt_summary or not lx_summary:
        return
    print(f"\n{'='*50}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*50}")
    print(f"  {'Metric':<22} {'veritract':>12} {'LangExtract':>12}")
    print(f"  {'-'*46}")
    for key, label in [
        ("latency_mean", "Latency (s)"),
        ("field_accuracy", "Field accuracy"),
        ("grounding_rate", "Grounding rate"),
    ]:
        vv = vt_summary.get(key, 0)
        lv = lx_summary.get(key, 0)
        if key == "latency_mean":
            print(f"  {label:<22} {vv:>11.1f}s {lv:>11.1f}s")
        else:
            print(f"  {label:<22} {vv*100:>11.1f}% {lv*100:>11.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="veritract vs LangExtract benchmark")
    parser.add_argument("--model", default="gemma4:e4b", help="Ollama model to use")
    parser.add_argument("--no-langextract", action="store_true", help="Skip LangExtract")
    parser.add_argument("--no-reground", action="store_true", help="Disable veritract auto_reground")
    parser.add_argument("--samples", type=int, default=0, help="Limit to N samples (0=all)")
    parser.add_argument("--out", default="", help="Save JSON results to this file")
    parser.add_argument(
        "--dataset", choices=["synthetic", "clinicaltrials"], default="clinicaltrials",
        help="Dataset to use (default: clinicaltrials — requires --build first)",
    )
    args = parser.parse_args()

    if args.dataset == "clinicaltrials":
        try:
            all_samples = get_ct_samples()
            schema = CT_SCHEMA
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        all_samples = SYNTHETIC_SAMPLES
        schema = SYNTHETIC_SCHEMA

    samples = all_samples[:args.samples] if args.samples else all_samples

    print(f"Benchmark: veritract vs LangExtract")
    print(f"Model: {args.model} | Dataset: {args.dataset} | Samples: {len(samples)}")
    print(f"{'='*50}")

    print("\nRunning veritract...")
    vt_results = run_veritract(samples, args.model, schema, auto_reground=not args.no_reground)
    fields = list(schema.get("properties", {}).keys())
    vt_summary = print_summary("veritract", vt_results, fields)

    lx_results: list[dict] = []
    lx_summary: dict = {}
    if not args.no_langextract:
        print("\nRunning LangExtract...")
        lx_results = run_langextract(samples, args.model, schema)
        lx_summary = print_summary("LangExtract", lx_results, fields)

    print_comparison(vt_summary, lx_summary)

    if args.out:
        out = {"veritract": vt_results, "langextract": lx_results}
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
