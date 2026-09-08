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
