from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from typing_extensions import TypedDict

from veritract.grounding import _DEFAULT_THRESHOLDS


class PipelineManifest(TypedDict):
    """Content-addressed record of everything that determines an extraction's output.

    ``manifest_id`` is a SHA-256 hash over every field below *except* ``manifest_id``
    and ``created_at``. Two runs with the same model digest, decoding params, prompt,
    schema, and thresholds share a ``manifest_id``. See ``docs/reproducibility.md``.
    """

    manifest_id: str
    veritract_version: str
    model_tag: str
    model_digest: str
    decoding: dict
    prompt_hash: str
    schema_hash: str
    thresholds: dict
    terminology_versions: dict
    rule_versions: dict
    created_at: str


class ManifestUnavailable(RuntimeError):
    """The environment cannot satisfy a manifest — the pinned model digest is absent or changed."""


def _sha(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _veritract_version() -> str:
    try:
        from importlib.metadata import version

        return version("veritract")
    except Exception:
        return "0+unknown"


def _decoding_options(llm) -> dict:
    if hasattr(llm, "decoding_options"):
        return dict(llm.decoding_options())
    return {
        k: getattr(llm, k)
        for k in ("temperature", "top_p", "seed")
        if getattr(llm, k, None) is not None
    }


def build_manifest(
    llm,
    schema: dict,
    *,
    prompt: str | None = None,
    thresholds: dict[str, int] | None = None,
    extra: dict | None = None,
) -> PipelineManifest:
    """Build a PipelineManifest describing how ``extract()`` would run with these settings.

    Args:
        llm: An ``LLMClient`` (or anything exposing ``model``, ``model_digest()``,
            and ``decoding_options()``).
        schema: The extraction JSON schema.
        prompt: Custom prompt, if any. Hashed; ``"default"`` sentinel when ``None``.
        thresholds: Grounding thresholds. Defaults to veritract's built-in
            ``{"abstract": 75, "fulltext": 85}`` — recorded resolved, as extraction sees them.
        extra: Optional ``{"terminology_versions": {...}, "rule_versions": {...}}`` merged
            into the manifest. Empty/omitted dicts do not perturb ``manifest_id``. Used by
            downstream packages (verichart) to pin terminology releases and business rules.
    """
    extra = extra or {}
    resolved_thresholds = (
        thresholds if thresholds is not None else dict(_DEFAULT_THRESHOLDS)
    )

    core = {
        "veritract_version": _veritract_version(),
        "model_tag": getattr(llm, "model", "unknown"),
        "model_digest": llm.model_digest(),
        "decoding": _decoding_options(llm),
        "prompt_hash": _sha(prompt) if prompt is not None else "default",
        "schema_hash": _sha(schema),
        "thresholds": resolved_thresholds,
        "terminology_versions": dict(extra.get("terminology_versions", {})),
        "rule_versions": dict(extra.get("rule_versions", {})),
    }
    return PipelineManifest(
        manifest_id=_sha(core),
        created_at=datetime.now(timezone.utc).isoformat(),
        **core,
    )
