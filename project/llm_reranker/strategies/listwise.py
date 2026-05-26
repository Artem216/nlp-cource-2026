from __future__ import annotations

import logging
from math import ceil
from typing import Sequence

from .base import BaseRerankStrategy
from .prompt_templates import render_prompt_template
from ..types import CandidateDocument, QueryExample, RerankResult


logger = logging.getLogger(__name__)


class ListwiseRankGPTRerankStrategy(BaseRerankStrategy):
    name = "listwise-rankgpt"

    def __init__(
        self,
        *,
        provider,
        prompt_language: str,
        query_max_chars: int,
        doc_max_chars: int,
        window_size: int,
        stride: int,
    ) -> None:
        super().__init__(
            provider=provider,
            prompt_language=prompt_language,
            query_max_chars=query_max_chars,
            doc_max_chars=doc_max_chars,
        )
        self.window_size = window_size
        self.stride = stride
        logger.info(
            "Listwise strategy window settings: window_size=%s, stride=%s",
            window_size,
            stride,
        )

    def rerank(
        self,
        query: QueryExample,
        documents: Sequence[CandidateDocument],
    ) -> RerankResult:
        if len(documents) <= 1:
            logger.debug(
                "Listwise rerank skipped: query_id=%s, documents=%s",
                query.query_id,
                len(documents),
            )
            return RerankResult(ordered_doc_ids=[document.doc_id for document in documents])

        ranking = list(documents)
        truncated_query = self._truncate_query(query.text)
        windows_processed = 0
        provider_calls = 0
        logger.debug(
            "Listwise rerank started: query_id=%s, documents=%s, query_chars=%s->%s, "
            "window_size=%s, stride=%s",
            query.query_id,
            len(documents),
            len(query.text),
            len(truncated_query),
            self.window_size,
            self.stride,
        )

        start = max(0, len(ranking) - self.window_size)
        while True:
            window_docs = ranking[start : start + self.window_size]
            if len(window_docs) > 1:
                logger.debug(
                    "Listwise window rerank: query_id=%s, start=%s, size=%s, doc_ids=%s",
                    query.query_id,
                    start,
                    len(window_docs),
                    [doc.doc_id for doc in window_docs],
                )
                system_prompt, user_prompt = self._build_prompts(
                    query_text=truncated_query,
                    documents=window_docs,
                )
                response = self.provider.generate_json(
                    cache_namespace=f"{self.name}:{query.query_id}:{start}:{','.join(doc.doc_id for doc in window_docs)}",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    schema=self._schema_for_window(window_docs),
                    metadata={
                        "query_id": query.query_id,
                        "window_start": start,
                        "doc_ids": [doc.doc_id for doc in window_docs],
                    },
                )
                ordered_ids = []
                if isinstance(response, dict):
                    ordered_ids = response.get("ordered_doc_ids", [])
                else:
                    logger.warning(
                        "Listwise provider response is not an object: query_id=%s, "
                        "window_start=%s, response_type=%s",
                        query.query_id,
                        start,
                        type(response).__name__,
                    )
                ranking[start : start + self.window_size] = self._reorder_window(
                    window_docs,
                    ordered_ids,
                )
                logger.debug(
                    "Listwise window applied: query_id=%s, start=%s, returned_ids=%s",
                    query.query_id,
                    start,
                    ordered_ids,
                )
                windows_processed += 1
                provider_calls += 1
            if start == 0:
                break
            start = max(0, start - self.stride)

        self._record(
            queries=1,
            provider_calls=provider_calls,
            documents_considered=len(documents),
            windows=windows_processed,
        )
        estimated_windows = (
            1
            if len(documents) <= self.window_size
            else ceil(max(0, len(documents) - self.window_size) / self.stride) + 1
        )
        logger.debug(
            "Listwise rerank finished: query_id=%s, ordered_doc_ids=%s, "
            "windows=%s/%s, provider_calls=%s",
            query.query_id,
            [document.doc_id for document in ranking],
            windows_processed,
            estimated_windows,
            provider_calls,
        )
        return RerankResult(
            ordered_doc_ids=[document.doc_id for document in ranking],
            metadata={
                "windows": windows_processed,
                "estimated_windows": estimated_windows,
            },
        )

    def stats(self) -> dict[str, int | str]:
        payload = super().stats()
        payload.update({"window_size": self.window_size, "stride": self.stride})
        return payload

    def _reorder_window(
        self,
        window_docs: Sequence[CandidateDocument],
        ordered_ids: Sequence[str],
    ) -> list[CandidateDocument]:
        by_id = {document.doc_id: document for document in window_docs}
        reordered: list[CandidateDocument] = []
        seen: set[str] = set()
        for doc_id in ordered_ids:
            if doc_id in by_id and doc_id not in seen:
                reordered.append(by_id[doc_id])
                seen.add(doc_id)
        for document in window_docs:
            if document.doc_id not in seen:
                reordered.append(document)
        return reordered

    def _build_prompts(
        self,
        *,
        query_text: str,
        documents: Sequence[CandidateDocument],
    ) -> tuple[str, str]:
        candidates = []
        for document in documents:
            candidates.append(
                "\n".join(
                    [
                        f"doc_id: {document.doc_id}",
                        f"candidate_rank: {document.original_rank + 1}",
                        "document:",
                        self._truncate_document(document.text),
                    ]
                )
            )
        system_prompt = render_prompt_template(
            self.name,
            self.prompt_language,
            "system",
        )
        user_prompt = render_prompt_template(
            self.name,
            self.prompt_language,
            "user",
            query_text=query_text,
            candidates="\n\n".join(candidates),
        )
        return system_prompt, user_prompt

    @staticmethod
    def _schema_for_window(documents: Sequence[CandidateDocument]) -> dict[str, object]:
        doc_ids = [document.doc_id for document in documents]
        return {
            "type": "object",
            "properties": {
                "ordered_doc_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": doc_ids},
                    "minItems": len(doc_ids),
                    "maxItems": len(doc_ids),
                }
            },
            "required": ["ordered_doc_ids"],
            "additionalProperties": False,
        }
