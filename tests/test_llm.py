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
