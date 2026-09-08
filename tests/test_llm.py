import pytest
from veritract.llm import LLMClient, MockLLM


def test_mock_llm_registered_response():
    llm = MockLLM()
    llm.register("extract sample", {"sample_size": "100"})
    result = llm.chat([{"role": "user", "content": "please extract sample size"}])
    assert result == {"sample_size": "100"}


def test_mock_llm_no_match_raises():
    llm = MockLLM()
    with pytest.raises(ValueError, match="no registered response"):
        llm.chat([{"role": "user", "content": "unrecognized prompt"}])


def test_mock_llm_multiple_registrations_first_match_wins():
    llm = MockLLM()
    llm.register("alpha", {"field": "a"})
    llm.register("alpha beta", {"field": "b"})
    # "alpha" matches first, so returns {"field": "a"}
    result = llm.chat([{"role": "user", "content": "alpha beta"}])
    assert result == {"field": "a"}


def test_mock_llm_schema_ignored():
    llm = MockLLM()
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    llm.register("test", {"x": "value"})
    result = llm.chat([{"role": "user", "content": "test"}], schema=schema)
    assert result == {"x": "value"}


def test_llm_client_options_stored():
    llm = LLMClient(temperature=0.0, top_p=0.9, seed=42)
    assert llm._options["temperature"] == 0.0
    assert llm._options["top_p"] == 0.9
    assert llm._options["seed"] == 42


def test_llm_client_no_options_by_default():
    llm = LLMClient()
    assert llm._options == {}


def test_llm_client_partial_options():
    llm = LLMClient(seed=7)
    assert llm._options == {"seed": 7}
    assert "temperature" not in llm._options


# --- model_digest (Task 1: pipeline manifest) ---


def test_mock_llm_model_digest_is_stable():
    llm = MockLLM()
    assert llm.model_digest() == llm.model_digest()
    assert llm.model_digest().startswith("sha256:")
    assert len(llm.model_digest()) == len("sha256:") + 64


def test_llm_client_model_digest_missing_model_raises():
    llm = LLMClient(model="veritract-nonexistent-model:v0")
    with pytest.raises(RuntimeError, match="veritract-nonexistent-model:v0"):
        llm.model_digest()


def test_llm_client_model_digest_resolves_present_model():
    """Against a real local Ollama model. Skips cleanly when Ollama is unavailable."""
    ollama = pytest.importorskip("ollama")
    try:
        listing = ollama.list()
    except Exception:
        pytest.skip("Ollama daemon not reachable")
    models = getattr(listing, "models", None) or (
        listing.get("models", []) if hasattr(listing, "get") else []
    )
    tags = {
        (m.get("model") if hasattr(m, "get") else None) or getattr(m, "model", None)
        for m in models
    }
    present = next((t for t in ("gemma3:1b", "gemma4:12b", "gemma4:e4b") if t in tags), None)
    if present is None:
        pytest.skip("no known model pulled")
    digest = LLMClient(model=present).model_digest()
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64
