# Reproducibility via PipelineManifest

## Overview

`PipelineManifest` lets you record *exactly how* an extraction was performed—model version, quantization, sampling parameters, prompt, schema, and thresholds—and later re-run the identical pipeline months or years later to verify that the same inputs produce the same outputs. This matters when you need to explain to an auditor where a value came from, re-extract a batch of documents after fixing a schema, or debug why a past extraction chose a particular grounding strategy.

## What the manifest pins

Every field is deterministic and immutable once the manifest is created:

| Field | Type | Purpose |
|---|---|---|
| `manifest_id` | SHA-256 string | Unique hash of the entire pipeline configuration. Collision-resistant. |
| `veritract_version` | str | Package version at extraction time. Needed to track schema/API changes. |
| `model_tag` | str | Ollama model tag (e.g. `"gemma4:e4b"`). Used to retrieve the model. |
| `model_digest` | str | Ollama content-address digest (`"sha256:<hex>"`). Pins the exact model manifest — weights, template, params — not just the tag. |
| `decoding` | dict | Sampling params forwarded to llama.cpp — a subset of `{"temperature", "top_p", "seed"}`, only the keys that were set. `{}` if none. |
| `prompt_hash` | str | SHA-256 of the custom prompt string. The sentinel `"default"` when no custom prompt was passed. |
| `schema_hash` | str | SHA-256 of the JSON schema (canonical, sorted keys). |
| `thresholds` | dict | Resolved grounding thresholds as extraction sees them — `{source_type: int}`, e.g. `{"abstract": 75, "fulltext": 85}`. |
| `terminology_versions` | dict | Empty in core veritract; populated by the downstream verichart package (e.g. `{"SNOMED-CT": "2026-03"}`). |
| `rule_versions` | dict | Empty in core veritract; populated by verichart (business-rule versions for its reasoning layer). |
| `created_at` | ISO 8601 string | Timestamp of manifest creation. NOT included in the `manifest_id` hash. |

**Key design:** Only deterministic, content-addressed fields contribute to the `manifest_id`. `created_at` is stored for audit trails but doesn't affect the hash, so manifests created at different times with identical configurations have identical `manifest_id` values.

## What `replay()` guarantees — and what it doesn't

### Guarantees

Given:
- A `PipelineManifest` from a past extraction
- The same input documents, supplied again as `{"text", "doc_id", "source_type"}` dicts
  (the manifest stores only hashes of the schema and prompt — never the documents)
- The same `schema` and `prompt` (verified against their stored hashes)
- The pinned model still available in Ollama at its content-addressed digest

You get:
- **Identical pipeline construction.** The prompt is built the same way, grounding thresholds are applied identically, verification mode is unchanged.
- **Stamped results.** Each `ExtractionResult` carries the same `manifest_id` as the original extraction.
- **No implicit changes.** veritract's internal logic at the version specified in the manifest is used; no features from newer versions.

### Does NOT guarantee

- **Identical LLM output.** Even with `temperature=0` and a fixed `seed`, the LLM may produce different text tokens if:
  - A different `llama.cpp` build is used (different C++ compiler flags, SIMD instructions, quantization library version).
  - The CPU/GPU running inference differs (floating-point math on different hardware can diverge in the last few bits of precision).
  - Ollama itself updates the llama.cpp version.
  
  In practice, with the same quantization and hardware, `seed=0` in llama.cpp produces near-identical output on repeated runs; edge cases (rare special tokens, numerical instability) account for the difference. **Veritract cannot guarantee bit-for-bit identical output**; it guarantees the *same pipeline is executed*.

- **Availability.** The manifest records `model_digest`, but does not store the model itself. If you discard the GGUF and later try to replay, Ollama cannot pull it back unless the tag still resolves to that exact digest. Models are not archived indefinitely on registries.

- **Collision immunity.** `manifest_id` is SHA-256, which is cryptographically strong, but SHA-256 collisions are theoretically possible (infeasible in practice). For forensic purposes, store the full manifest, not just the hash.

## Why a hosted LLM API cannot satisfy this contract

Hosted APIs like OpenAI, Anthropic, or Gemini cannot provide reproducible extraction because:

1. **Non-deterministic output.** Even with `temperature=0`, API providers do not guarantee identical output on re-runs. Internally, requests may be:
   - Routed to different hardware.
   - Load-balanced across different model instances.
   - Processed through mixture-of-experts (MoE) routing that varies per request.
   - Subject to batching strategies that affect numerics.
   
   Providers document this: identical requests to the same endpoint may produce different outputs.

2. **Model versions are not pinned.** A model name like `"gpt-4o"` is an endpoint alias that the provider updates on their schedule. The model serving that alias changes silently every few weeks or months. If you extract in 2026 and replay in 2028, the endpoint may be retired or serve a different model entirely.

3. **No content address.** You cannot hash "the model" or archive the exact weights. When `llm.model_digest()` is called against an API client, there is no digest to read—the model is opaque and versioned only by a name the provider controls.

veritract's reproducibility feature is **Ollama-native** precisely because Ollama:
- Pins models by SHA-256 content address (the GGUF file hash is immutable).
- Runs inference locally (no hidden batching or routing; you see exactly which hardware is used).
- Forwards sampling `seed` through to llama.cpp, giving you source-level control.
- Lets you archive quantized models locally.

## Usage example

Extract once, save the manifest and the inputs:

```python
import json
from veritract import extract, build_manifest, LLMClient

llm = LLMClient(model="gemma4:e4b", temperature=0.0, seed=42)

schema = {
    "type": "object",
    "properties": {
        "drug": {"type": "string"},
        "sample_size": {"type": "string"},
        "outcome": {"type": "string"},
    },
    "required": ["drug", "sample_size", "outcome"],
}

# Build the manifest FIRST, then pass it to extract() so results are stamped.
# If you use a custom `prompt` or `thresholds`, pass the SAME values here.
manifest = build_manifest(llm, schema)

inputs = [{"text": document_text, "doc_id": "doc_001", "source_type": "text"}]
result = extract(inputs[0]["text"], schema, llm,
                 doc_id=inputs[0]["doc_id"], source_type=inputs[0]["source_type"],
                 manifest=manifest)

# Persist the manifest and the inputs — the manifest stores only hashes, never the documents.
with open("extraction_log.json", "w") as f:
    json.dump({"manifest": manifest, "inputs": inputs}, f, indent=2)

print(result.manifest_id)  # == manifest["manifest_id"]
```

Replay the extraction months later:

```python
import json
from veritract import replay

with open("extraction_log.json") as f:
    log = json.load(f)

# Supply the SAME schema (and prompt, if one was used) — replay verifies their hashes.
# llm=None → an LLMClient is built from manifest["model_tag"] + manifest["decoding"].
results = replay(log["manifest"], log["inputs"], schema=schema)

for r in results:
    assert r.manifest_id == log["manifest"]["manifest_id"]
    print(r.extracted)
```

`replay()` checks, in order:
- the passed `schema` hashes to `manifest["schema_hash"]` (else `ValueError`);
- the passed `prompt` hashes to `manifest["prompt_hash"]` (else `ValueError`);
- `llm.model_digest()` resolves and equals `manifest["model_digest"]` (else `ManifestUnavailable`).

It then runs `extract()` per input with the manifest's thresholds, and stamps every result
with the manifest's id.

## `ManifestUnavailable`

Raised when `replay()` cannot reconstruct the extraction pipeline. Specifically:

```python
class ManifestUnavailable(RuntimeError):
    """The environment cannot satisfy a manifest — the pinned model digest is absent or changed."""
```

**When it's raised:**
- `llm.model_digest()` cannot resolve — the model tag is not present in Ollama and the daemon
  has nothing to hash (raised as `ManifestUnavailable` wrapping the underlying `RuntimeError`).
- `llm.model_digest()` resolves but the digest differs from `manifest["model_digest"]` — the tag
  still exists but now points to a different model manifest (the registry was updated, or it was
  re-pulled at a newer version).

**What to do:**
1. **Check what's available:** `ollama list` to see which models are cached.
2. **Try to pull the exact tag:** `ollama pull gemma4:e4b` will attempt to pull from the Ollama registry. If the tag still resolves to the same digest, the pull succeeds and `replay()` can proceed.
3. **If the tag no longer resolves to the same digest:** The model genuinely cannot be reproduced. This is expected on long time horizons; model registries deprecate versions. Document this in your audit log: *"Model gemma4:e4b was used for extraction on 2026-09-08; this exact model is no longer available as of 2028-11-15."* and treat the extraction as non-reproducible from that point forward.

If you need to preserve reproducibility indefinitely, manually archive the GGUF file:

```bash
# Locate the cached GGUF on disk
find ~/.ollama/models -name "*.gguf" | grep gemma4

# Copy it to cold storage
cp /path/to/gemma4_e4b.gguf s3://my-archive/models/gemma4_e4b_sha256_abc123.gguf
```

Then, to replay years later:
1. Pull the archived GGUF back locally.
2. Re-serve it under the original tag (or update the manifest tag).
3. Run `replay()`.

This is outside veritract's scope but is feasible if reproducibility is critical.
