from __future__ import annotations

import logging
from typing import Sequence

from .base import BaseRerankStrategy
from .prompt_templates import render_prompt_template
from ..types import CandidateDocument, QueryExample, RerankResult


logger = logging.getLogger(__name__)


PAIRWISE_SCHEMA = {
    "type": "object",
    "properties": {
        "winner_doc_id": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
        }
    },
    "required": ["winner_doc_id"],
    "additionalProperties": False,
}


class PairwisePRPRerankStrategy(BaseRerankStrategy):
    name = "pairwise-prp"

    def rerank(
        self,
        query: QueryExample,
        documents: Sequence[CandidateDocument],
    ) -> RerankResult:
        comparison_cache: dict[tuple[str, str], str | None] = {}
        comparison_count = 0
        provider_calls = 0
        truncated_query = self._truncate_query(query.text)
        logger.debug(
            "Pairwise rerank started: query_id=%s, documents=%s, query_chars=%s->%s",
            query.query_id,
            len(documents),
            len(query.text),
            len(truncated_query),
        )

        def compare(left: CandidateDocument, right: CandidateDocument) -> bool:
            nonlocal comparison_count, provider_calls
            comparison_count += 1
            if left.doc_id == right.doc_id:
                return True

            first, second = sorted((left, right), key=lambda document: document.doc_id)
            cache_key = (first.doc_id, second.doc_id)
            if cache_key not in comparison_cache:
                system_prompt, user_prompt = self._build_prompts(
                    query_text=truncated_query,
                    first=first,
                    second=second,
                )
                response = self.provider.generate_json(
                    cache_namespace=f"{self.name}:{query.query_id}:{first.doc_id}:{second.doc_id}",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    schema=PAIRWISE_SCHEMA,
                    metadata={
                        "query_id": query.query_id,
                        "doc_ids": [first.doc_id, second.doc_id],
                    },
                )
                winner_doc_id = None
                if isinstance(response, dict):
                    winner_doc_id = response.get("winner_doc_id")
                if winner_doc_id not in {first.doc_id, second.doc_id}:
                    logger.debug(
                        "Pairwise provider returned tie/invalid winner: query_id=%s, "
                        "left=%s, right=%s, winner=%r",
                        query.query_id,
                        first.doc_id,
                        second.doc_id,
                        winner_doc_id,
                    )
                    winner_doc_id = None
                comparison_cache[cache_key] = winner_doc_id
                provider_calls += 1
                logger.debug(
                    "Pairwise comparison completed: query_id=%s, first=%s, second=%s, "
                    "winner=%s, provider_calls=%s",
                    query.query_id,
                    first.doc_id,
                    second.doc_id,
                    winner_doc_id,
                    provider_calls,
                )

            winner_doc_id = comparison_cache[cache_key]
            if winner_doc_id is None:
                return left.original_rank <= right.original_rank
            return winner_doc_id == left.doc_id

        ordered = self._merge_sort(list(documents), compare)
        self._record(
            queries=1,
            provider_calls=provider_calls,
            documents_considered=len(documents),
            comparisons=comparison_count,
        )
        logger.debug(
            "Pairwise rerank finished: query_id=%s, ordered_doc_ids=%s, "
            "comparisons=%s, provider_calls=%s",
            query.query_id,
            [document.doc_id for document in ordered],
            comparison_count,
            provider_calls,
        )
        return RerankResult(
            ordered_doc_ids=[document.doc_id for document in ordered],
            metadata={"comparisons": comparison_count},
        )

    def _merge_sort(
        self,
        documents: list[CandidateDocument],
        compare,
    ) -> list[CandidateDocument]:
        if len(documents) <= 1:
            return documents
        middle = len(documents) // 2
        left = self._merge_sort(documents[:middle], compare)
        right = self._merge_sort(documents[middle:], compare)
        merged: list[CandidateDocument] = []
        left_index = 0
        right_index = 0
        while left_index < len(left) and right_index < len(right):
            if compare(left[left_index], right[right_index]):
                merged.append(left[left_index])
                left_index += 1
            else:
                merged.append(right[right_index])
                right_index += 1
        merged.extend(left[left_index:])
        merged.extend(right[right_index:])
        return merged

    def _build_prompts(
        self,
        *,
        query_text: str,
        first: CandidateDocument,
        second: CandidateDocument,
    ) -> tuple[str, str]:
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
            first_doc_id=first.doc_id,
            first_text=self._truncate_document(first.text),
            second_doc_id=second.doc_id,
            second_text=self._truncate_document(second.text),
        )
        return system_prompt, user_prompt
