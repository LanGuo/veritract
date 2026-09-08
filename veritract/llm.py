from __future__ import annotations

import json
import time
import ollama
from typing import Any


class LLMClient:
    """Ollama wrapper with retry and constrained JSON output via GBNF grammar."""

    def __init__(
        self,
        model: str = "gemma4:e4b",
        max_retries: int = 3,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ):
        self.model = model
        self.max_retries = max_retries
        self._options: dict[str, Any] = {
            k: v for k, v in {
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
            }.items() if v is not None
        }

    def model_digest(self) -> str:
        """Return the Ollama content-address digest for ``self.model``.

        The value is ``"sha256:<64 hex>"`` — the digest of the model's Ollama
        manifest, which transitively covers the weights, template, and
        parameters. It is a stable content address for pinning an extraction run
        so it can be reproduced later (see ``docs/reproducibility.md``).

        Raises:
            RuntimeError: if the Ollama daemon is unreachable or ``self.model``
                is not present locally. A pinned model that cannot be resolved is
                exactly the condition ``replay()`` must surface.
        """
        try:
            listing = ollama.list()
        except Exception as e:
            raise RuntimeError(
                f"cannot resolve model digest for {self.model!r}: Ollama not reachable ({e})"
            ) from e
        models = getattr(listing, "models", None)
        if models is None:
            models = listing.get("models", []) if hasattr(listing, "get") else []
        for m in models:
            tag = (m.get("model") if hasattr(m, "get") else None) or getattr(m, "model", None)
            if tag == self.model:
                digest = (m.get("digest") if hasattr(m, "get") else None) or getattr(m, "digest", None)
                if digest:
                    digest = str(digest)
                    return digest if digest.startswith("sha256:") else f"sha256:{digest}"
        raise RuntimeError(
            f"model {self.model!r} not found in Ollama; cannot pin its digest"
        )

    def decoding_options(self) -> dict[str, Any]:
        """Sampling params forwarded to Ollama (temperature / top_p / seed), if set."""
        return dict(self._options)

    def chat(self, messages: list[dict], schema: dict | None = None, think: bool = False) -> dict:
        """Call Ollama and return parsed JSON (when schema given) or {"text": ...}.

        Args:
            messages: Chat messages. Each dict may include an ``images`` key with
                a list of base64-encoded PNG strings for multimodal calls.
            schema: JSON schema dict passed as Ollama ``format=`` for constrained
                decoding. llama.cpp masks invalid tokens at sampling time.
            think: Enable extended reasoning (Gemma 4 thinking mode).
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
                if schema:
                    kwargs["format"] = schema
                if think:
                    kwargs["think"] = True
                if self._options:
                    kwargs["options"] = self._options
                response = ollama.chat(**kwargs)
                content = response["message"]["content"]
                return json.loads(content) if schema else {"text": content}
            except json.JSONDecodeError as e:
                last_error = e
                raw_content = content  # already bound before json.loads raised
                messages = messages + [
                    {"role": "assistant", "content": raw_content},
                    {"role": "user", "content": "Your response was not valid JSON. Reply with valid JSON only."},
                ]
                time.sleep(2 ** attempt)
            except Exception as e:
                last_error = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM failed after {self.max_retries} attempts: {last_error}")


class MockLLM:
    """Deterministic LLM stub for tests. Register canned responses by prompt substring."""

    def __init__(
        self,
        *,
        model: str = "mock",
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ):
        self._responses: list[tuple[str, dict]] = []
        self.model = model
        self._options: dict[str, Any] = {
            k: v for k, v in {
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
            }.items() if v is not None
        }

    def register(self, prompt_contains: str, response: dict) -> None:
        self._responses.append((prompt_contains, response))

    def model_digest(self) -> str:
        """Fixed sentinel digest so manifests built with MockLLM are deterministic."""
        return "sha256:" + "0" * 64

    def decoding_options(self) -> dict[str, Any]:
        return dict(self._options)

    def chat(self, messages: list[dict], schema: dict | None = None, think: bool = False) -> dict:
        full_text = " ".join(m.get("content", "") for m in messages)
        for substr, resp in self._responses:
            if substr in full_text:
                return resp
        raise ValueError(f"MockLLM: no registered response for prompt: {full_text[:120]}")
