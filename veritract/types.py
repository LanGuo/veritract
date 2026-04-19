from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing_extensions import TypedDict


class Span(TypedDict):
    doc_id: str | None
    source_type: str
    char_start: int
    char_end: int
    text: str
    provenance_type: Literal["direct", "paraphrased", "inferred"]


class GroundedField(TypedDict):
    value: str
    span: Span | None
    confidence: float


class QuarantinedField(TypedDict):
    field_name: str
    value: str
    reason: str


@dataclass
class ExtractionResult:
    extracted: dict[str, GroundedField]
    quarantined: list[QuarantinedField]

    @property
    def provenance(self) -> list[Span]:
        return [f["span"] for f in self.extracted.values() if f["span"] is not None]
