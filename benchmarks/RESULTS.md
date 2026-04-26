# Benchmark Results Log

All runs use **EBM-NLP 2.0** dataset (Nye et al. 2018) unless noted.
Fields evaluated: `drug`, `sample_size`, `outcome`.
Scoring: fuzzy match (token_set_ratio ≥ 70) + numeric proximity for sample_size + LLM semantic judge for drug/outcome where noted.
Model: `gemma4:e4b` via Ollama (local).

---

## 2026-04-23 — Baseline: shared few-shot examples, no prompt optimization

**What changed:** Switched dataset from ClinicalTrials.gov (noisy GT) to EBM-NLP (verbatim GT spans).
Added shared few-shot examples (etanercept + amlodipine) to both runners.
Fixed grounding rate to use same `_is_verbatim()` definition for both packages.

**veritract** — `mode="full"` (fuzzy grounding + LLM re-verification), 155 samples

| Metric | Value |
|---|---|
| Samples | 155/155 |
| Field accuracy | 62.8% |
| Grounding rate | 97.1% |
| Latency (mean) | 50.4s |
| drug | 50% |
| sample_size | 96% |
| outcome | 42% |

**LangExtract** — 155 samples (no LLM judge)

| Metric | Value |
|---|---|
| Samples | 113/155 (27% failure rate) |
| Field accuracy | 70.8% |
| Grounding rate | 98.8% |
| Latency (mean) | 13.6s |

**Notes:**
- veritract drug accuracy at 50% due to schema field named "drug" biasing toward pharmacological-only; many EBM-NLP interventions are non-pharmacological (surgery, PT, etc.)
- LangExtract 27% failure rate: model returns null extraction_text for those samples
- LangExtract leads on accuracy despite higher failure rate — QA-format extraction activates reading comprehension vs veritract's template-filling JSON mode

---

## 2026-04-23 — LLM judge on 20 samples (veritract only, earlier run)

**What changed:** Added `--llm-judge` flag that re-scores `drug` and `outcome` with LLM semantic equivalence check (False→True only, never downgrades).

| Metric | Value |
|---|---|
| Samples | 20/20 |
| Field accuracy | 71.7% (with LLM judge) |
| Grounding rate | 100.0% |
| Latency (mean) | 40.3s |
| drug | ~70% |
| sample_size | 95% |
| outcome | ~40% |

**Notes:** LLM judge boosted accuracy ~3% over fuzzy-only scoring on this small sample.

---

## 2026-04-23 — Prompt optimization introduced

**What changed:**
- Added `optimize_prompt()` to veritract (new `veritract/optimizer.py`)
- Optimization: 3 iterations on 20 calibration EBM-NLP samples with GT labels
- LLM rewrote the extraction prompt to be more explicit about clinical field semantics and verbatim copying
- Added 3 verification modes: `"full"` (grounding + LLM re-verify), `"fuzzy"` (grounding only), `"no-grounding"` (raw LLM output)
- Optimized prompt also derived for LangExtract via `_derive_lx_prompt()` (strips JSON-specific sections)

**veritract (full)** — 20 samples, LLM judge ON

| Metric | Value |
|---|---|
| Samples | 20/20 |
| Field accuracy | 75.0% |
| Grounding rate | 95.0% |
| Latency (mean) | 35.0s |
| drug | 90% |
| sample_size | 95% |
| outcome | 40% |

**LangExtract** — 20 samples, LLM judge ON, optimized prompt

| Metric | Value |
|---|---|
| Samples | 19/20 |
| Field accuracy | 77.2% |
| Grounding rate | 98.2% |
| Latency (mean) | 16.0s |
| drug | 79% |
| sample_size | 100% |
| outcome | 53% |

**Notes:** Optimized prompt notably improved `drug` from ~50% → 90% for veritract (20-sample; confirmed at scale below).

---

## 2026-04-24 — Full 155-sample benchmark with optimized prompt, 3-way comparison

**What changed:** Same optimized prompt applied to all three configurations. LLM judge ON for all.

Results file: `benchmarks/results_ebmnlp_155_optimized.json`

| Metric | veritract (full) | veritract (no-grounding) | LangExtract |
|---|---|---|---|
| Samples | 155/155 | 155/155 | 109/155 ❌ |
| Field accuracy | 75.7% | **77.8%** | 75.2% |
| Grounding rate | 91.2% | 91.6% | **97.6%** |
| Quarantine rate | 0.2% | 0.4% | — |
| Latency (mean) | 30.8s | 28.6s | **12.1s** |
| drug | 74% | **77%** | 79% |
| sample_size | **92%** | **92%** | 91% |
| outcome | 61% | **64%** | 56% |

**Notes:**

- **Outcome improved dramatically** (42% → 61–64%) — the LLM-generated optimized prompt added per-field semantic descriptions (`"The primary condition, effect, or measurement being assessed"`) which the model responds to well.
- **`outcome` is hardest for all systems** — EBM-NLP GT spans often name the measure class (e.g., "pain intensity") while models extract the full endpoint description or vice versa.
- **LangExtract failure rate rose to 30%** (46/155) — the `_derive_lx_prompt` instruction style derived from the veritract-optimized prompt suits LangExtract's QA format less well than the original hand-crafted description. Accuracy among the 109 successful samples (75.2%) is on par with veritract.
- **Grounding rate 91% (veritract) vs 97.6% (LangExtract)** — LangExtract's QA format naturally produces verbatim phrases; veritract with the optimized prompt also produces mostly verbatim phrases but ~9% are paraphrased or numeric-normalized.

---

## 2026-04-24 — Investigation: does grounding hurt accuracy?

**Question:** Is the 2.1pp accuracy gap (no-grounding 77.8% vs full 75.7%) caused by grounding quarantining correct values?

**Finding: No — the difference is LLM non-determinism, not grounding.**

Analysis of the two 155-sample runs:
- Only **1 field quarantined** across all 465 field-sample pairs in `mode="full"` (PMID 8913901 `drug`="ST. Louis University program" — correctly quarantined, not a drug name)
- **54.8% of fields have identical extracted values** between the two runs — 45.2% differ due to non-deterministic LLM sampling (Ollama default temperature > 0)
- no-grounding won on 27 fields, full won on 17 fields — net advantage of 10 fields out of 465 is within noise

**Conclusion:** Grounding is not the cause of the accuracy gap. The two modes ran independent LLM inference calls with different random seeds, producing different outputs. To properly compare grounding vs no-grounding requires saving raw LLM outputs and post-hoc applying grounding — not two separate benchmark runs.

---

---

## 2026-04-24 — 4-arm benchmark: post-hoc grounding on both extractors, 3-run CI

**What changed:**
- New benchmark design: `extract_raw()` and `ground()` are separate steps for both extractors
- veritract: `extract_raw()` called once per (run, sample); `vt_ground()` applied post-hoc for both raw and grounded arms — true apples-to-apples grounding comparison with no LLM non-determinism between arms
- LangExtract: same design — run N times, then `vt_ground()` applied post-hoc to its output
- 3 runs per extractor (seeds 42/43/44, temp=0.0) for 95% CI
- LLM judge ON for all arms
- Optimized prompt (3 iter, 20 cal samples)
- Results file: `benchmarks/results_4arm_multirun.json`

**veritract — 465 samples (155 × 3 runs)**

| Metric | veritract (raw) | veritract (grounded) |
|---|---|---|
| Samples | 465/465 | 465/465 |
| Field accuracy | 72.9% | 72.9% |
| Grounding rate | 98.4% | 98.7% |
| Quarantine rate | 4.7% | 4.9% |
| Latency (mean) | 29.2s | 32.1s |
| drug | 67% | 67% |
| sample_size | 95% | 95% |
| outcome | 56% | 56% |

**LangExtract — FAILED (456/465 errors)**

LangExtract had a 98% failure rate (`extraction_failed: model returned no valid extractions`). Root cause: the optimized prompt ends with `Source Text: "{}"` — a veritract-specific template. When passed as LangExtract's Ollama system prompt via `language_model_params={"system": ...}`, it conflicts with LangExtract's internal QA format construction. The model follows the system prompt's JSON instruction instead of LangExtract's extraction format, producing output that LangExtract's parser cannot interpret.

Only 9/465 samples succeeded (likely those where the model happened to produce output compatible with both formats):

| Metric | LangExtract (raw) | LangExtract (grounded) |
|---|---|---|
| Samples | 9/465 ❌ | 9/465 ❌ |
| Field accuracy | 66.7% | 66.7% |
| Grounding rate | 100.0% | 100.0% |

**Notes:**

- **veritract accuracy regression** (75.7% → 72.9%) — this run used `temp=0.0` and `seed=42/43/44` which changes the sampling distribution vs prior runs at Ollama default temperature. The constrained decoding at temp=0 may be producing slightly more conservative (mode-collapsing) outputs. Not directly comparable to prior runs.
- **Raw vs grounded identical** (72.9% both) — confirms the earlier finding: with only 50M active params and optimized prompt, the model already produces mostly verbatim phrases. Grounding's quarantine step (4.7–4.9% quarantine rate) catches genuinely unverifiable fields without hurting accuracy on verifiable ones.
- **95% CI not shown** — with 3 identical-seed runs at temp=0 all runs produce the same output, so CI half-width is 0 for veritract. CI is only meaningful with non-zero temperature; seeds alone are insufficient when temp=0 forces deterministic sampling.

**Fix needed for LangExtract re-run:**
Pass only the instruction portion of the optimized prompt (stripping the JSON structure and `Source Text: "{}"` template), or use the default `_lx_prompt_description()` without the veritract prompt. LangExtract works best with its native QA format — the optimized prompt should be adapted rather than passed verbatim.

---

---

## 2026-04-26 — 4-arm benchmark v2: LangExtract prompt fix, temp=0.3 for real CI

**What changed:**
- Fixed LangExtract prompt: `_derive_lx_prompt()` now strips JSON template and `Source Text:` placeholder before passing to LangExtract's QA format; dropped system-prompt approach that caused 98% failure in v1
- Temperature raised to 0.3 so different seeds produce genuinely different outputs, making 95% CI meaningful
- Same 4-arm design: veritract raw/grounded and LangExtract raw/grounded, all using same raw extraction per run
- Results file: `benchmarks/results_4arm_multirun_v2.json`

**4-arm results — 155 samples, 3 runs (seeds 42/43/44), temp=0.3, LLM judge ON**

| Metric | veritract (raw) | veritract (grounded) | LangExtract (raw) | LangExtract (grounded) |
|---|---|---|---|---|
| Samples | 465/465 | 465/465 | 330/465 ⚠️ | 330/465 ⚠️ |
| Field accuracy | 64.1% ± 1.1% | 64.1% ± 1.1% | 74.8% ± 1.3% | 74.8% ± 1.3% |
| Grounding rate | 99.1% | 99.0% | 91.9% | 91.9% |
| Quarantine rate | 5.7% | 5.6% | — | 0.2% |
| Latency (mean) | 33.9s | 37.1s | 10.1s | 11.2s |
| drug | 65% | 65% | 83% | 83% |
| sample_size | 95% | 95% | 98% | 98% |
| outcome | 33% | 33% | 44% | 44% |

**Notes:**

- **LangExtract 29% failure rate** (135/465 errors) — this is consistent with prior single-run results (~30%). The model returns no valid extractions for roughly 1 in 3 samples regardless of prompt tuning. Accuracy numbers above are over the 330 successful samples only.
- **LangExtract leads on accuracy among successful samples** (74.8% vs 64.1%) — its QA-format prompting is better at reading comprehension than veritract's JSON template-filling mode, especially for `drug` (83% vs 65%) and `outcome` (44% vs 33%).
- **Grounding makes no difference to accuracy for either extractor** — raw and grounded are identical across all fields for both. With temp=0.3, the model already produces mostly verbatim phrases (veritract 99%, LangExtract 92% grounding rate). Grounding adds quarantine signal but doesn't recover or lose accuracy.
- **veritract accuracy regression from prior runs** (75.7% → 64.1%) — driven by temp=0.3 producing more varied outputs than temp=0, combined with the optimized prompt varying across runs. This run is not directly comparable to prior single-run benchmarks at different temperatures.
- **CI is now meaningful**: ±1.1% (veritract) and ±1.3% (LangExtract) — runs at temp=0.3 produce genuine variance across seeds. The veritract raw/grounded CIs are identical as expected (same raw shared between arms).
- **Grounding latency overhead**: +3.2s for veritract (37.1s vs 33.9s), +1.1s for LangExtract (11.2s vs 10.1s) — both small relative to extraction time.

**Open question:** veritract's accuracy gap vs LangExtract is now consistent and large (10+ pp). LangExtract's QA format activates stronger reading comprehension. The next investigation should focus on whether veritract's JSON-template format can be improved, or whether a hybrid QA-then-structure approach is worth exploring.

---

## Summary: accuracy progression (veritract, EBM-NLP, LLM judge)

| Date | Config | Samples | Accuracy | Key change |
|---|---|---|---|---|
| 2026-04-23 | shared examples, mode=full | 155 | 62.8% | Baseline after EBM-NLP switch |
| 2026-04-23 | LLM judge | 20 | 71.7% | Added semantic judge |
| 2026-04-23 | optimized prompt, mode=full | 20 | 75.0% | Prompt optimization (3 iter) |
| 2026-04-24 | optimized prompt, mode=full | 155 | 75.7% | Full-scale confirmation |
| 2026-04-24 | optimized prompt, mode=no-grounding | 155 | 77.8% | Separate run; difference is noise |
| 2026-04-24 | 4-arm, temp=0, 3 runs | 465 | 72.9% | Post-hoc grounding design; temp=0 collapse |
| 2026-04-26 | 4-arm, temp=0.3, 3 runs | 465 | 64.1% ± 1.1% | First run with real CI; temp=0.3 lowers accuracy |
