from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QueryExample:
    query_id: str
    text: str


@dataclass(frozen=True)
class CandidateDocument:
    doc_id: str
    text: str
    original_rank: int


@dataclass(frozen=True)
class RerankResult:
    ordered_doc_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
