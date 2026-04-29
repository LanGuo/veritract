# veritract Benchmark: Design and Results

## Overview

Head-to-head evaluation of **veritract** vs **LangExtract** on clinical information extraction from biomedical abstracts. Both systems use the same local Ollama model (`gemma4:e4b`) and the same dataset, with results scored against expert-annotated ground truth.

---

## Dataset

**EBM-NLP 2.0** (Nye et al., 2018)  
Expert-annotated token spans for clinical trial abstracts across three structured fields:

| Field | Description | GT source |
|---|---|---|
| `drug` | Intervention name (pharmacological or non-pharmacological) | Interventions hierarchical label 3 |
| `sample_size` | Number of participants enrolled | Participants hierarchical label 3 |
| `outcome` | Primary outcome measure | Outcomes starting spans (any non-zero label) |

Ground truth is stored as `all_gt_spans` — the full set of unique annotated spans per field. Scoring accepts a prediction as correct if it matches **any** span in this set, not just the first annotation. This matters because EBM-NLP sometimes annotates multiple valid spans for a single field (e.g., both "234 patients" and "two hundred thirty-four patients" for the same abstract).

Cache file: `benchmarks/ebmnlp_cache.json` (git-ignored).  
Samples used: **155** (those with at least one non-empty GT field after filtering).

---

## Extractors

### veritract

veritract's extraction pipeline:

```
optimize_prompt() → extract_raw() → ground(mode="full")
```

1. **Prompt optimization** (`optimize_prompt()`): iterative refinement over a calibration subset. Each iteration extracts from calibration samples, scores against GT, then asks the LLM to suggest prompt improvements. Returns the prompt with the highest calibration accuracy across all iterations.
2. **Extraction** (`extract_raw()`): single LLM call with GBNF-constrained JSON decoding (Ollama `format=` param). Invalid tokens masked at sampling time — model cannot produce malformed JSON.
3. **Grounding** (`ground(mode="full")`): two-stage source verification:
   - Fuzzy match (rapidfuzz `token_set_ratio ≥ 70` + `partial_ratio ≥ 85` + acronym detection) — marks fields as `direct` or `paraphrased`
   - LLM re-verification for fuzzy failures — marks verified fields as `inferred`
   - Quarantine: fields that pass neither stage are surfaced separately with the raw extracted value and failure reason

Each result row stores per-field **spans** (`char_start`, `char_end`, `provenance_type`, `confidence`) and **quarantine** entries (`field`, `raw_value`, `reason`).

### LangExtract

LangExtract uses a QA-style prompting format where the model answers "What is the [field]?" for each field in sequence. This activates reading comprehension rather than template-filling.

Veritract grounding is applied **post-hoc** to LangExtract's output:

```
LangExtract.extract() → RawExtractionResult wrapper → vt_ground(mode="full")
```

This gives LangExtract the same span storage and quarantine signal as veritract.

**Prompt derivation for LangExtract** (`_derive_lx_prompt()`): the veritract-optimized prompt is adapted before passing to LangExtract's `prompt_description`. Sections stripped:
- `Return JSON ...` lines (causes model to output JSON instead of LangExtract's extraction format)
- `Examples:` / `Text:` / `Source Text:` template sections
- Any `{}` placeholder lines

The remaining instruction portion (field definitions, verbatim-copy rules) is passed as `prompt_description`, preserving LangExtract's native QA format.

---

## Run Parameters

| Parameter | Value |
|---|---|
| Model | `gemma4:e4b` via Ollama (local) |
| Temperature | `0.3` (non-zero for real CI across seeds) |
| Seeds | 42, 43, 44 (3 runs per extractor) |
| Prompt optimization | 3 iterations, 20 calibration samples, GT-supervised |
| LLM judge | ON — re-scores `drug` and `outcome` via semantic equivalence check (False → True only) |
| Samples | 155 |
| Total rows | 465 per extractor (155 × 3 runs) |

**Temperature note:** `temp=0` produces fully deterministic output — seeds alone do not create variance, making CI meaningless. `temp=0.3` produces ~20% variance across seeds, yielding meaningful 95% CI (half-width ~1–1.5pp observed).

---

## Scoring

### Fuzzy field scoring

- **`sample_size`**: numeric proximity first (|predicted_N − gt_N| / gt_N ≤ 15%), then fuzzy fallback for written-out numbers.
- **`drug`, `outcome`**: `token_set_ratio ≥ 70` OR `partial_ratio ≥ 85` OR acronym match (e.g. "OC" → "Oral contraceptive", checked bidirectionally).
- All string comparisons are case-insensitive.

### LLM judge

For `drug` and `outcome`, a second LLM call re-evaluates fields scored False by fuzzy matching:

```
Field: outcome
Acceptable GT values: "pain intensity | mean daily nasal symptom scores"
Model extracted: "nasal symptom score"
→ CORRECT if semantically equivalent to ANY GT value
```

Rules: correct if same clinical concept, more specific version, or abbreviation of any GT value; wrong if different concept or empty.

### Multi-span GT

`_score_field()` accepts `str | list[str]` for the expected value. When `all_gt_spans` is provided (EBM-NLP), the field scores True on the first matching span (deduped iteration). The LLM judge formats the full list as `"span1 | span2 | ..."` and instructs the model to match any.

### Grounding metrics (new in v3)

Beyond accuracy, each run reports:

- **Provenance breakdown**: count of `direct` / `paraphrased` / `inferred` fields across all grounded fields
- **Quarantine precision**: fraction of quarantined fields that were actually wrong (true signal vs false alarm)
- **Quarantine recall**: fraction of wrong fields that were caught by quarantine

---

## Intermediate Result Saving

Full results are saved to JSON after each complete run: `benchmarks/results_v3_2arm.json` (git-ignored). Each row in the JSON has:

```json
{
  "id": "PMID...",
  "run": 0,
  "seed": 42,
  "latency_extraction": 28.5,
  "latency_grounding": 3.2,
  "latency": 31.7,
  "extracted": {"drug": "budesonide Turbuhaler", "sample_size": "284"},
  "spans": {
    "drug": {"char_start": 44, "char_end": 65, "provenance_type": "direct", "confidence": 100.0},
    "sample_size": {"char_start": 210, "char_end": 237, "provenance_type": "paraphrased", "confidence": 87.3},
    "outcome": null
  },
  "quarantined": [{"field": "outcome", "raw_value": "...", "reason": "fuzzy_failed"}],
  "accuracy": {"drug": true, "sample_size": true, "outcome": false},
  "field_acc": 0.667,
  "grounding_rate": 0.667,
  "quarantine_rate": 0.333,
  "error": null
}
```

The top-level JSON has keys `"veritract"` and `"langextract"`, each a flat list of rows.

---

## Prior Benchmark History

See `benchmarks/result_logs.md` for full run history and notes on methodology evolution (4-arm → 2-arm redesign, temp=0 determinism issue, LangExtract prompt fixes).

---

## v3 Results — 2026-04-28

**Run parameters:** 155 samples × 3 runs (seeds 42/43/44), temp=0.3, LLM judge ON, optimized prompt (3 iter, 20 cal samples).  
Results file: `benchmarks/results_v3_2arm.json`

### Accuracy metrics

| Metric | veritract | LangExtract |
|---|---|---|
| Samples | 465/465 ✅ | 465/465 ✅ |
| Field accuracy | **87.7% ± 0.3%** | 84.7% ± 1.9% |
| `drug` | 69% | 69% |
| `sample_size` | **97%** | 94% |
| `outcome` | **97%** | 92% |

### Latency

| Stage | veritract | LangExtract |
|---|---|---|
| Extraction only | 28.9s | **5.5s** |
| Verification cost | +5.4s | +0.2s |
| **Total (mean)** | **34.3s** | **5.7s** |

Extraction compares `extract_raw()` (veritract, GBNF-constrained JSON decoding) vs `lx.extract()` (LangExtract, free-form QA). The 5× extraction gap is driven by GBNF token-masking overhead and LangExtract generating shorter per-field answers.

Verification cost for veritract (+5.4s mean, up to 53s) is dominated by LLM re-verification calls on quarantined fields (136 extra LLM calls across the run). LangExtract's grounding is near-zero (+0.2s) because no fields are quarantined — grounding is just fuzzy matching, which is instant.

### Completeness metrics

Every extracted field must end up in one of three states:

| State | veritract | LangExtract |
|---|---|---|
| Grounded (has verified span) | 90.3% | 85.7% |
| Quarantined (extracted but unverifiable) | 9.7% | 0.0% |
| Missing (never extracted) | 0.0% | ~14.3% |
| **Total** | **100%** | **100%** |

veritract's GBNF grammar forces all schema fields to be present in every output — nothing is ever missing. If the model doesn't know the answer it produces a string that fails grounding and gets quarantined. LangExtract's QA format can simply not return a field; those absent fields reduce the grounding rate without appearing as quarantined.

### Provenance breakdown

When a field passes grounding it is labelled with how it was verified:

- **`direct`** — extracted value is a case-insensitive exact substring of the source. The model copied the phrase verbatim.  
  *Example: source has "budesonide Turbuhaler 400 microg/day" and exactly that string was extracted.*

- **`paraphrased`** — fuzzy match passed (token_set_ratio ≥ 70 or partial_ratio ≥ 85) but not an exact substring. The model condensed, abbreviated, or normalized the phrase.  
  *Example: extracted "234 patients" when source says "two hundred thirty-four patients"; or extracted "budesonide" when source has "budesonide Turbuhaler."*

- **`inferred`** — fuzzy match failed but LLM re-verification confirmed it. Typically abbreviations, unit conversions, or semantic paraphrases that fuzzy string matching can't catch.  
  *Example: extracted "ACE inhibitor" when source uses "enalapril." Rare — the verbatim-copy prompt keeps this near zero.*

| Provenance | veritract | LangExtract |
|---|---|---|
| `direct` | 652 (52%) | 401 (34%) |
| `paraphrased` | 602 (48%) | 791 (66%) |
| `inferred` | 5 (0%) | 3 (0%) |

veritract produces roughly equal direct/paraphrased (52/48%). LangExtract skews heavily toward paraphrased (66%): its QA format produces shorter, condensed answers that often don't appear as exact substrings — the model extracts a sub-phrase rather than the full span, or normalizes phrasing slightly.

### Verification quality (veritract only)

LangExtract has 0% quarantine not because its outputs are more trustworthy, but because the fields it does return all pass fuzzy grounding — LangExtract's QA format naturally produces verbatim phrases that are easy to locate in the source. The fields it can't answer it simply skips (→ missing), so there is nothing for quarantine to catch.

For veritract, quarantine is the signal that an extracted value could not be anchored to the source text. Treating quarantine as a binary classifier ("this field is wrong"):

| Quarantine metric | Value |
|---|---|
| Precision | **100%** — 136/136 quarantined fields were genuinely wrong per GT |
| Recall | **79%** — 136 of 172 total wrong fields were caught; 36 escaped |

**Precision 100%** means the quarantine signal is perfectly reliable: if veritract flags a field, it is wrong. No correct extractions were quarantined.

**Recall 79%** means 21% of wrong fields (36 out of 172) were not quarantined — they passed grounding but were semantically wrong. These are "grounded hallucinations": the model extracted a real phrase from the source text (so grounding accepted it) but the wrong one for that field. Examples: a sample size from a different study arm, a secondary outcome instead of the primary, a drug name from the background section referring to a comparator. Grounding can only check whether a value *appears* in the source, not whether it is the *right* value for the field.

### Key findings

- **veritract leads on accuracy** (87.7% vs 84.7%), a reversal from v2 (64.1% vs 74.8%). Both systems improved dramatically.

- **LangExtract 0% failure rate**, fixed from a consistent ~29–30% failure rate across all prior runs. Root cause: `_derive_lx_prompt()` was passing `"Return JSON with exactly these fields: ..."` through to LangExtract's `prompt_description`, causing the model to output JSON instead of LangExtract's extraction format. Stripping "return json" lines resolved this.

- **Dramatic outcome improvement** — veritract `outcome` 33% → 97%; LangExtract 44% → 92% vs v2. Primary cause: multi-span GT scoring (`all_gt_spans`). EBM-NLP annotates multiple valid spans per field; prior runs scored against only the first annotation, missing valid extractions.

- **Drug accuracy tied at 69%** — EBM-NLP labels non-pharmacological interventions (surgery, physical therapy, behavioral programs) under the same field as drugs. Models associate "drug" with pharmacological agents and fail on non-drug interventions.

- **veritract CI is tighter** (±0.3% vs ±1.9%) — GBNF-constrained decoding produces more consistent structured outputs across seeds than LangExtract's free-form QA.

### Optimized prompt (v3)

```
You are an expert information extractor. Your task is to extract specific pieces of
information from the provided medical text and return the results in a JSON format.
You must copy the extracted values **verbatim** from the source text. If a field
cannot be found, return an empty string for that field.

Extract the following fields:
  * drug: [Verbatim extraction of the drug name.]
  * sample_size: [Verbatim extraction of the study population size.]
  * outcome: [Verbatim extraction of the primary health outcome or measurement.]
```

For LangExtract, `_derive_lx_prompt()` strips the "Return JSON" instruction, leaving only the field definitions and verbatim-copy rule.
