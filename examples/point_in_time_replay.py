"""
Point-in-time replay: extract once, persist a PipelineManifest, reproduce later.

Run with Ollama up and a small model pulled (e.g. `ollama pull gemma3:1b`):

    python examples/point_in_time_replay.py

See docs/reproducibility.md for what replay() guarantees and what it doesn't.
"""
import json
import tempfile
from pathlib import Path

from veritract import LLMClient, build_manifest, extract, replay

MODEL = "gemma3:1b"

SCHEMA = {
    "type": "object",
    "properties": {
        "drug": {"type": "string"},
        "sample_size": {"type": "string"},
        "primary_outcome": {"type": "string"},
    },
    "required": ["drug", "sample_size", "primary_outcome"],
}

DOC = (
    "In a randomized controlled trial, 248 patients with type 2 diabetes received "
    "either metformin 500mg twice daily or placebo. The primary outcome was HbA1c "
    "reduction at 12 months."
)

# temperature=0 + fixed seed → near-deterministic decoding (see docs/reproducibility.md)
llm = LLMClient(model=MODEL, temperature=0.0, seed=42)

# 1. Build the manifest FIRST, then stamp the extraction with it.
manifest = build_manifest(llm, SCHEMA)
inputs = [{"text": DOC, "doc_id": "trial:001", "source_type": "text"}]

result = extract(
    inputs[0]["text"], SCHEMA, llm,
    doc_id=inputs[0]["doc_id"], source_type=inputs[0]["source_type"],
    manifest=manifest,
)
print(f"extracted   : { {k: v['value'] for k, v in result.extracted.items()} }")
print(f"manifest_id : {result.manifest_id}")

# 2. Persist manifest + inputs (the manifest holds only hashes, never the document).
log_path = Path(tempfile.gettempdir()) / "veritract_replay_log.json"
log_path.write_text(json.dumps({"manifest": manifest, "inputs": inputs}, indent=2))
print(f"saved       : {log_path}")

# 3. ...time passes... reload and replay.
log = json.loads(log_path.read_text())
replayed = replay(log["manifest"], log["inputs"], schema=SCHEMA)  # llm=None → built from manifest

r = replayed[0]
assert r.manifest_id == manifest["manifest_id"]
same = {k: v["value"] for k, v in r.extracted.items()} == {
    k: v["value"] for k, v in result.extracted.items()
}
print(f"replayed    : { {k: v['value'] for k, v in r.extracted.items()} }")
print(f"values match: {same}")
