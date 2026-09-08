import pytest

from veritract.manifest import build_manifest, PipelineManifest
from veritract.llm import MockLLM

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}


def test_manifest_id_is_deterministic():
    llm = MockLLM()
    m1 = build_manifest(llm, SCHEMA)
    m2 = build_manifest(llm, SCHEMA)
    assert m1["manifest_id"] == m2["manifest_id"]
    assert len(m1["manifest_id"]) == 64  # sha256 hex


def test_manifest_id_changes_with_prompt():
    llm = MockLLM()
    base = build_manifest(llm, SCHEMA)
    changed = build_manifest(llm, SCHEMA, prompt="a very different prompt")
    assert base["manifest_id"] != changed["manifest_id"]


def test_manifest_id_changes_with_schema():
    llm = MockLLM()
    base = build_manifest(llm, SCHEMA)
    changed = build_manifest(llm, {"type": "object", "properties": {"a": {"type": "string"}}})
    assert base["manifest_id"] != changed["manifest_id"]


def test_manifest_id_changes_with_thresholds():
    llm = MockLLM()
    base = build_manifest(llm, SCHEMA)
    changed = build_manifest(llm, SCHEMA, thresholds={"text": 99})
    assert base["manifest_id"] != changed["manifest_id"]


def test_manifest_id_changes_with_model_digest():
    a = MockLLM()
    b = MockLLM()
    b.model_digest = lambda: "sha256:" + "a" * 64
    assert build_manifest(a, SCHEMA)["manifest_id"] != build_manifest(b, SCHEMA)["manifest_id"]


def test_empty_extra_does_not_change_id():
    llm = MockLLM()
    base = build_manifest(llm, SCHEMA)
    same = build_manifest(llm, SCHEMA, extra={"terminology_versions": {}, "rule_versions": {}})
    assert base["manifest_id"] == same["manifest_id"]


def test_extra_populates_and_changes_id():
    llm = MockLLM()
    base = build_manifest(llm, SCHEMA)
    with_term = build_manifest(
        llm, SCHEMA, extra={"terminology_versions": {"SNOMED-CT": "2026-03"}}
    )
    assert with_term["terminology_versions"] == {"SNOMED-CT": "2026-03"}
    assert with_term["rule_versions"] == {}
    assert with_term["manifest_id"] != base["manifest_id"]


def test_created_at_not_in_manifest_id():
    llm = MockLLM()
    m1 = build_manifest(llm, SCHEMA)
    m2 = build_manifest(llm, SCHEMA)
    assert m1["manifest_id"] == m2["manifest_id"]  # equal despite differing created_at
    assert "created_at" in m1 and m1["created_at"]


def test_decoding_params_captured():
    llm = MockLLM(temperature=0.0, seed=42)
    m = build_manifest(llm, SCHEMA)
    assert m["decoding"] == {"temperature": 0.0, "seed": 42}


def test_decoding_params_change_id():
    base = build_manifest(MockLLM(), SCHEMA)
    seeded = build_manifest(MockLLM(seed=42), SCHEMA)
    assert base["manifest_id"] != seeded["manifest_id"]


def test_model_tag_recorded():
    m = build_manifest(MockLLM(model="gemma4:e4b"), SCHEMA)
    assert m["model_tag"] == "gemma4:e4b"
    assert m["model_digest"].startswith("sha256:")


def test_prompt_hash_is_default_sentinel_when_none():
    m = build_manifest(MockLLM(), SCHEMA)
    assert m["prompt_hash"] == "default"


def test_manifest_shape():
    m = build_manifest(MockLLM(), SCHEMA)
    for key in (
        "manifest_id", "veritract_version", "model_tag", "model_digest", "decoding",
        "prompt_hash", "schema_hash", "thresholds", "terminology_versions",
        "rule_versions", "created_at",
    ):
        assert key in m


# --- replay (Task 4) ---


def _replay_llm():
    llm = MockLLM(model="gemma3:1b", seed=42)
    llm.register("Extract the following fields", {"a": "alpha", "b": "beta"})
    return llm


def test_replay_reproduces_identical_results():
    from veritract.manifest import replay
    llm = _replay_llm()
    manifest = build_manifest(llm, SCHEMA)
    inputs = [{"text": "here alpha and also beta", "doc_id": "d1", "source_type": "text"}]
    r1 = replay(manifest, inputs, schema=SCHEMA, llm=llm)
    r2 = replay(manifest, inputs, schema=SCHEMA, llm=llm)
    assert r1[0].manifest_id == manifest["manifest_id"]
    assert r1[0].extracted.keys() == r2[0].extracted.keys()
    assert all(
        r1[0].extracted[k]["value"] == r2[0].extracted[k]["value"] for k in r1[0].extracted
    )


def test_replay_returns_one_result_per_input():
    from veritract.manifest import replay
    llm = _replay_llm()
    manifest = build_manifest(llm, SCHEMA)
    inputs = [
        {"text": "alpha beta one", "doc_id": "d1", "source_type": "text"},
        {"text": "alpha beta two", "doc_id": "d2", "source_type": "text"},
    ]
    out = replay(manifest, inputs, schema=SCHEMA, llm=llm)
    assert len(out) == 2


def test_replay_rejects_mismatched_schema():
    from veritract.manifest import replay
    manifest = build_manifest(_replay_llm(), SCHEMA)
    with pytest.raises(ValueError, match="schema"):
        replay(
            manifest, [{"text": "x", "doc_id": None, "source_type": "text"}],
            schema={"type": "object", "properties": {"z": {"type": "string"}}},
            llm=_replay_llm(),
        )


def test_replay_rejects_mismatched_prompt():
    from veritract.manifest import replay
    manifest = build_manifest(_replay_llm(), SCHEMA)  # prompt=None → "default"
    with pytest.raises(ValueError, match="prompt"):
        replay(
            manifest, [{"text": "x", "doc_id": None, "source_type": "text"}],
            schema=SCHEMA, prompt="a custom prompt", llm=_replay_llm(),
        )


def test_replay_raises_manifest_unavailable_on_digest_mismatch():
    from veritract.manifest import replay, ManifestUnavailable
    manifest = dict(build_manifest(_replay_llm(), SCHEMA))
    manifest["model_digest"] = "sha256:" + "f" * 64
    with pytest.raises(ManifestUnavailable):
        replay(
            manifest, [{"text": "x", "doc_id": None, "source_type": "text"}],
            schema=SCHEMA, llm=_replay_llm(),
        )


def test_replay_without_llm_unresolvable_model_raises_manifest_unavailable():
    from veritract.manifest import replay, ManifestUnavailable
    llm = MockLLM(model="veritract-not-a-real-model:v0")
    llm.register("Extract the following fields", {"a": "alpha", "b": "beta"})
    manifest = build_manifest(llm, SCHEMA)
    with pytest.raises(ManifestUnavailable):
        replay(
            manifest, [{"text": "alpha beta", "doc_id": None, "source_type": "text"}],
            schema=SCHEMA,  # llm=None → real LLMClient, model_digest() fails
        )
