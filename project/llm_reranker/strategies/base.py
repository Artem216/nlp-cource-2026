from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Sequence

from ..providers import BaseLLMProvider
from ..types import CandidateDocument, QueryExample, RerankResult
from ..utils import truncate_text


logger = logging.getLogger(__name__)


class BaseRerankStrategy(ABC):
    name = "strategy"

    def __init__(
        self,
        *,
        provider: BaseLLMProvider,
        prompt_language: str,
        query_max_chars: int,
        doc_max_chars: int,
    ) -> None:
        self.provider = provider
        self.prompt_language = prompt_language
        self.query_max_chars = query_max_chars
        self.doc_max_chars = doc_max_chars
        self._lock = threading.Lock()
        self._stats: dict[str, int] = {
            "queries": 0,
            "provider_calls": 0,
            "documents_considered": 0,
        }
        logger.info(
            "Rerank strategy initialized: strategy=%s, prompt_language=%s, "
            "query_max_chars=%s, doc_max_chars=%s",
            self.name,
            prompt_language,
            query_max_chars,
            doc_max_chars,
        )

    @abstractmethod
    def rerank(
        self,
        query: QueryExample,
        documents: Sequence[CandidateDocument],
    ) -> RerankResult:
        raise NotImplementedError

    def stats(self) -> dict[str, Any]:
        with self._lock:
            payload = dict(self._stats)
        payload.update(
            {
                "strategy": self.name,
                "prompt_language": self.prompt_language,
                "query_max_chars": self.query_max_chars,
                "doc_max_chars": self.doc_max_chars,
            }
        )
        logger.debug("Rerank strategy stats requested: %s", payload)
        return payload

    def _record(self, **updates: int) -> None:
        with self._lock:
            for key, value in updates.items():
                self._stats[key] = self._stats.get(key, 0) + value

    def _truncate_query(self, text: str) -> str:
        return truncate_text(text, self.query_max_chars)

    def _truncate_document(self, text: str) -> str:
        return truncate_text(text, self.doc_max_chars)
