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
constrained decoding  →  fuzzy grounding  →  LLM re-verification  →  typed provenance + quarantine
```

See [comparables.md](comparables.md) for a full comparison with Instructor, LangExtract, NuExtract, GLiNER, and others.

## License

MIT
