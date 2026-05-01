# veritract

**Verified extraction — truth-anchored LLM structured output.**

LLM extraction where every field earns its place: fuzzy-matched to source text,
LLM-re-verified if fuzzy fails, quarantined if neither passes. Typed provenance
on every field. Works with any Ollama-served model.

## Prerequisites

**Python 3.10+** and **[Ollama](https://ollama.com)** running locally.

1. Install Ollama: https://ollama.com/download  
2. Pull a model (gemma4 recommended for best results, gemma3 for lower memory):

```bash
ollama pull gemma4:e4b    # ~9 GB — best accuracy
ollama pull gemma3:12b    # ~8 GB — good balance
ollama pull gemma3:4b     # ~3 GB — fastest, lower accuracy
```

3. Verify Ollama is running:

```bash
ollama list
```

Ollama must be running before you call `LLMClient`. Start it with `ollama serve` if needed.

## Install

veritract is not yet on PyPI. Install from source:

```bash
git clone https://github.com/LanGuo/veritract.git
cd veritract
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e .                  # core
pip install -e '.[pdf]'           # with PDF + figure support (adds docling)
pip install -e '.[dev]'           # with test dependencies
```

**With PDF extraction support** — adds [docling](https://github.com/docling-project/docling) which handles text, tables, and scanned pages via OCR.

## Quick start

```python
from veritract import extract, LLMClient

llm = LLMClient(model="gemma3:4b")

schema = {
    "type": "object",
    "properties": {
        "sample_size": {"type": "string"},
        "intervention": {"type": "string"},
        "primary_outcome": {"type": "string"},
    },
    "required": ["sample_size", "intervention", "primary_outcome"],
}

result = extract(text=document_text, schema=schema, llm=llm)

# Grounded fields — verified against source
for field, gf in result.extracted.items():
    print(field, gf["value"], gf["span"]["provenance_type"])

# Fields that couldn't be verified — preserved for human review
for qf in result.quarantined:
    print(qf["field_name"], qf["value"], qf["reason"])
```

## Provenance types

| Type | Meaning |
|---|---|
| `direct` | Exact substring — value appears verbatim in source |
| `paraphrased` | Fuzzy match above threshold |
| `inferred` | LLM confirmed; fuzzy matching failed (e.g. abbreviation, unit conversion) |
| quarantined | Neither fuzzy nor LLM could verify — surfaced for human review |

## How it works

```
(optional) prompt optimization → constrained decoding → (optional) fuzzy grounding → (optional) LLM re-verification → typed provenance + quarantine
```

Each stage is independently accessible:

```python
from veritract import extract_raw, ground, optimize_prompt

# Stage 1: LLM call only — returns RawExtractionResult
raw = extract_raw(text, schema, llm)

# Stage 2: grounding only — reuse same raw with different strategies
result_full   = ground(raw, llm, mode="full")        # fuzzy + LLM re-verify
result_fuzzy  = ground(raw, llm, mode="fuzzy")       # fuzzy only, no LLM re-verify
result_none   = ground(raw, llm, mode="no-grounding") # raw values, no verification
```

Separating the two stages means you pay for LLM inference once and can apply
multiple grounding strategies — or compare them — without re-running the model.

## Multimodal (image) extraction

Pass images alongside text for models that support vision (e.g. `gemma4:e4b`):

```python
from veritract import extract, load_images_b64

images = load_images_b64(["figure1.png", "figure2.png"])
result = extract(caption_text, schema, llm, images=images)
```

When images are provided, veritract prepends an image-first preamble to the
prompt so the model processes visual content before reading the extraction
instructions — following best practice for vision-language attention.

For extracting from PDF figures, ground against the **full document text** (not
just the caption) so values visible in a figure but discussed in the paper body
can be verified. See [`examples/extract_figures.py`](examples/extract_figures.py).

## Verification modes

| Mode | Grounding | LLM re-verify | When to use |
|---|---|---|---|
| `"full"` (default) | ✓ fuzzy | ✓ on quarantined | Highest fidelity, provenance on every field |
| `"fuzzy"` | ✓ fuzzy | ✗ | Faster; still catches hallucinations |
| `"no-grounding"` | ✗ | ✗ | Speed/latency benchmarking; raw LLM output |

## Prompt optimization

```python
from veritract import optimize_prompt

# Unsupervised: score by grounding rate
best_prompt = optimize_prompt(examples, schema, llm, n_iter=3)

# Supervised: score by accuracy against ground-truth labels
best_prompt = optimize_prompt(
    examples, schema, llm,
    n_iter=5,
    ground_truth=[{"sample_size": "120", "intervention": "aspirin", ...}, ...],
    seed=42,
)

# Use the optimized prompt for extraction
result = extract(text, schema, llm, prompt=best_prompt)
```

`optimize_prompt` runs iterative prompt refinement: extract → score → ask the LLM
to suggest improvements → repeat. Returns the prompt with the highest score across
all iterations. Benchmarks show 10–20pp accuracy gains on clinical NLP tasks after
3–5 iterations.

## Reproducible inference

```python
llm = LLMClient(
    model="gemma4:e4b",
    temperature=0.0,   # near-deterministic output
    top_p=0.9,
    seed=42,
)
```

Sampling parameters are forwarded to Ollama at call time. Use `temperature=0.0`
and a fixed `seed` for reproducible benchmarks or CI pipelines.

## LLMClient

```python
LLMClient(
    model="gemma4:e4b",   # any Ollama model tag
    max_retries=3,        # auto-retry with JSON correction on parse failure
    temperature=None,     # float, e.g. 0.0 for near-deterministic
    top_p=None,           # float
    seed=None,            # int
)
```

`LLMClient.chat()` uses Ollama's GBNF-constrained decoding when a JSON schema is
provided — invalid tokens are masked at sampling time, so the model physically
cannot produce malformed JSON. This is stronger than post-hoc parsing or
regex-based repair.

## API reference

| Symbol | Description |
|---|---|
| `extract(text, schema, llm, *, mode, prompt, examples, images, doc_id, source_type, thresholds)` | One-call extraction + grounding |
| `extract_raw(text, schema, llm, *, prompt, examples, images, doc_id, source_type, max_text_chars)` | LLM call only → `RawExtractionResult` |
| `extract_pdf(path, schema, llm, *, chunk_size, chunk_overlap, mode, prompt, examples, thresholds)` | Extract from a PDF file via docling; requires `pip install 'veritract[pdf]'` |
| `ground(raw, llm, *, mode, thresholds)` | Grounding only on a `RawExtractionResult` |
| `optimize_prompt(examples, schema, llm, *, n_iter, n_sample, ground_truth, seed)` | Iterative prompt refinement |
| `load_images_b64(paths)` | Load image files as base64 PNG for multimodal extraction |
| `LLMClient(model, max_retries, temperature, top_p, seed)` | Ollama wrapper with retry + GBNF |
| `MockLLM()` | Deterministic stub for tests |
| `ExtractionResult` | Dataclass: `extracted: dict[str, GroundedField]`, `quarantined: list[QuarantinedField]`, `.provenance` |
| `RawExtractionResult` | Dataclass: `fields: dict[str, str]`, `garbage`, `source_text`, `doc_id`, `source_type` |
| `GroundedField` | TypedDict: `value`, `span: Span \| None`, `confidence` |
| `QuarantinedField` | TypedDict: `field_name`, `value`, `reason` |
| `Span` | TypedDict: `doc_id`, `source_type`, `char_start`, `char_end`, `text`, `provenance_type` |

## Comparison with alternatives

Most structured extraction tools stop at getting the LLM to return valid JSON. veritract's differentiator is the grounding layer: every extracted value is traced back to a character span in the source, and values that can't be verified are quarantined rather than silently returned. This matters when downstream consumers need to trust individual fields — not just the overall schema shape.

| | veritract | Instructor | LangExtract | NuExtract | GLiNER |
|---|---|---|---|---|---|
| **Backend** | Ollama (local) | Any LLM API | Ollama / Gemini | Local HF model | Local HF model |
| **Constrained JSON decoding** | ✓ GBNF token masking | ✓ (provider-dependent) | ✓ | ✓ | — (span extraction) |
| **Source grounding / provenance** | ✓ char spans, typed | ✗ | ✗ | ✗ | ✓ (span-native) |
| **Quarantine on unverifiable fields** | ✓ | ✗ | ✗ | ✗ | ✗ |
| **Separate extract / ground stages** | ✓ | ✗ | ✗ | ✗ | — |
| **Prompt optimization** | ✓ iterative, supervised | ✗ | ✗ | ✗ | ✗ |
| **Multimodal (image) input** | ✓ | provider-dependent | ✗ | ✗ | ✗ |
| **Fully local / air-gapped** | ✓ | ✗ (API required) | ✓ | ✓ | ✓ |
| **Schema-driven** | ✓ JSON Schema | ✓ Pydantic | class-based | ✓ | fixed NER types |

**Instructor** is the most popular choice for schema-validated LLM output but has no notion of whether extracted values actually appear in the source — hallucinated values pass through silently.

**LangExtract** uses a QA-style prompt format that naturally produces verbatim spans, giving high grounding rates, but provides no quarantine path and has a ~30% failure rate on some models. Its grounding is implicit rather than verified.

**NuExtract** is a fine-tuned extraction model with good throughput but no source verification and no mechanism for handling fields the model can't confidently extract.

**GLiNER** is span-extraction native (always grounded by construction) but is limited to predefined NER-style entity types — it can't handle arbitrary schema fields or multi-token structured values like sample sizes with units.

veritract's niche is **schema-flexible extraction with an explicit trust signal per field**: the provenance type (`direct` / `paraphrased` / `inferred` / quarantined) lets downstream code treat high-confidence and uncertain fields differently, rather than accepting or rejecting the entire extraction.

## Benchmark

Evaluated on **EBM-NLP 2.0** (155 expert-annotated clinical abstracts, fields: `drug`, `sample_size`, `outcome`), head-to-head against LangExtract using `gemma4:e4b` via Ollama. 3 runs × 155 samples, temp=0.3, optimized prompt (3 iter), LLM semantic judge on `drug` and `outcome`.

| | veritract | LangExtract |
|---|---|---|
| Field accuracy | **87.7% ± 0.3%** | 84.7% ± 1.9% |
| Extraction latency | 28.9s | **5.5s** |
| Verification cost | +5.4s | +0.2s |
| Grounded (has span) | **90.3%** | 85.7% |
| Quarantined | 9.7% | 0%† |
| Missing (not extracted) | 0% | ~14.3% |
| Quarantine precision | **100%** | — |
| Quarantine recall | 79% | — |

†LangExtract skips fields it can't answer; veritract's GBNF grammar always emits all schema fields, so unconfident extractions surface as quarantined rather than missing.

Quarantine precision 100%: every quarantined field was genuinely wrong. Quarantine recall 79%: the remaining 21% of wrong fields were "grounded hallucinations" — real phrases extracted from the source text but for the wrong field.

See [`benchmarks/benchmark.md`](benchmarks/benchmark.md) for full methodology, provenance breakdown, and run-by-run history.

## License

MIT
