"""
Test: extract architecture + training regime from LLM technical reports.
Shows raw LLM output vs grounded output side by side.
"""
import sys
from veritract import extract_pdf, LLMClient
from veritract.extraction import extract_raw, ground
from veritract.pdf import _chunk_text, _merge_raw_results
from pathlib import Path

llm = LLMClient(model="gemma3:12b", temperature=0.0, seed=42)

schema = {
    "type": "object",
    "properties": {
        "architecture_type": {"type": "string"},
        "total_parameters": {"type": "string"},
        "active_parameters_per_token": {"type": "string"},
        "context_length": {"type": "string"},
        "pretraining_token_count": {"type": "string"},
        "pretraining_data_sources": {"type": "string"},
        "pretraining_optimizer_and_tricks": {"type": "string"},
        "midtraining_or_continued_pretraining": {"type": "string"},
        "supervised_finetuning": {"type": "string"},
        "alignment_and_rl_method": {"type": "string"},
    },
    "required": [
        "architecture_type",
        "total_parameters",
        "active_parameters_per_token",
        "context_length",
        "pretraining_token_count",
        "pretraining_data_sources",
        "pretraining_optimizer_and_tricks",
        "midtraining_or_continued_pretraining",
        "supervised_finetuning",
        "alignment_and_rl_method",
    ],
}

PROMPT = """You are extracting facts from an LLM technical report.
Rules:
- Copy the exact verbatim phrase from the text. Do not paraphrase, abbreviate, or synthesise.
- If a field is not present in this specific excerpt, return an empty string. Do NOT use prior knowledge.

Fields:
* architecture_type: Model architecture family and variant — dense transformer, Mixture-of-Experts (MoE), etc. Include layer count, attention type (GQA, MHA, MLA, sliding window), and any structural notes if stated.
* total_parameters: Total parameter count as stated in the text (e.g. "27 billion parameters", "671B total").
* active_parameters_per_token: MoE only — number of parameters activated per token per forward pass as stated. Empty string for dense models.
* context_length: Maximum context window length as stated (tokens).
* pretraining_token_count: Total number of tokens used for pretraining as stated.
* pretraining_data_sources: Data mixture for pretraining — datasets, domains, or percentages as stated.
* pretraining_optimizer_and_tricks: Optimizer name, LR schedule, batch size, or training stability techniques as stated.
* midtraining_or_continued_pretraining: Any training phase between pretraining and SFT as stated — long-context extension, domain annealing, etc. Empty string if not mentioned.
* supervised_finetuning: SFT stage description as stated — data sources, scale, curriculum.
* alignment_and_rl_method: Post-SFT alignment method as stated — RLHF, DPO, GRPO, PPO, reward model details."""

pdfs = {
    "Gemma 3":    "/tmp/llm_reports/gemma3.pdf",
    "Kimi K2.5":  "/tmp/llm_reports/kimi_k2_5.pdf",
    "Qwen 3":     "/tmp/llm_reports/qwen3.pdf",
}

CHUNK_SIZE = 8000
CHUNK_OVERLAP = 500

for name, path in pdfs.items():
    print(f"\n{'='*60}")
    print(f"  {name}  —  raw LLM extractions per chunk")
    print('='*60)

    from docling.document_converter import DocumentConverter
    conv = DocumentConverter()
    doc = conv.convert(str(path))
    full_text = doc.document.export_to_markdown()
    doc_id = Path(path).name
    chunks = _chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  PDF→markdown: {len(full_text):,} chars | {len(chunks)} chunks")

    raw_results = []
    for i, (chunk_text, offset) in enumerate(chunks):
        raw = extract_raw(chunk_text, schema, llm, prompt=PROMPT, doc_id=doc_id, source_type="pdf")
        # Show only non-empty, non-generic extractions
        hits = {k: v for k, v in raw.fields.items()
                if v and v.lower() not in ("details not provided", "not provided", "n/a")}
        if hits:
            print(f"\n  chunk {i+1}/{len(chunks)} (offset {offset:,}):")
            for field, val in hits.items():
                print(f"    {field}: {val[:120]}")
        raw_results.append(raw)

    print(f"\n  — Merging + grounding —")
    merged = _merge_raw_results(raw_results, full_text=full_text, doc_id=doc_id)
    result = ground(merged, llm, mode="fuzzy")

    print(f"\n  GROUNDED fields:")
    for field, gf in result.extracted.items():
        if gf["value"] and gf["value"].lower() not in ("details not provided", "not provided", "n/a", ""):
            ptype = gf["span"]["provenance_type"] if gf["span"] else "no-span"
            print(f"    [{ptype}] {field}: {gf['value'][:120]}")

    # Non-empty quarantined (ignore empty-string quarantines)
    real_q = [q for q in result.quarantined if q["value"]]
    if real_q:
        print(f"\n  QUARANTINED (non-empty, unverifiable):")
        for q in real_q:
            print(f"    {q['field_name']}: {q['value'][:80]!r} — {q['reason']}")

print("\nDone.")
