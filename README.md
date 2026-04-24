# veritract

**Verified extraction — truth-anchored LLM structured output.**

LLM extraction where every field earns its place: fuzzy-matched to source text,
LLM-re-verified if fuzzy fails, quarantined if neither passes. Typed provenance
on every field. Works with any Ollama-served model.

## Install

```bash
pip install veritract
```

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
| `extract_raw(text, schema, llm, *, prompt, examples, images, doc_id, source_type)` | LLM call only → `RawExtractionResult` |
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

See [comparables.md](comparables.md) for a full comparison with Instructor, LangExtract, NuExtract, GLiNER, and others.

## License

MIT
