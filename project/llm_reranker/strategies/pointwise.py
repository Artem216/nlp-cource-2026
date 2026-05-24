from __future__ import annotations

import logging
from typing import Any, Sequence

from .base import BaseRerankStrategy
from .prompt_templates import render_prompt_template
from ..types import CandidateDocument, QueryExample, RerankResult


logger = logging.getLogger(__name__)


POINTWISE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "grade": {"type": "integer", "minimum": 0, "maximum": 3},
                },
                "required": ["doc_id", "grade"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


class PointwiseGradedRerankStrategy(BaseRerankStrategy):
    name = "pointwise-graded"

    def rerank(
        self,
        query: QueryExample,
        documents: Sequence[CandidateDocument],
    ) -> RerankResult:
        truncated_query = self._truncate_query(query.text)
        logger.debug(
            "Pointwise rerank started: query_id=%s, documents=%s, "
            "query_chars=%s->%s",
            query.query_id,
            len(documents),
            len(query.text),
            len(truncated_query),
        )
        payload_lines = []
        for document in documents:
            payload_lines.append(
                "\n".join(
                    [
                        f"doc_id: {document.doc_id}",
                        f"candidate_rank: {document.original_rank + 1}",
                        "document:",
                        self._truncate_document(document.text),
                    ]
                )
            )
        system_prompt, user_prompt = self._build_prompts(
            query_text=truncated_query,
            document_payload="\n\n".join(payload_lines),
        )
        response = self.provider.generate_json(
            cache_namespace=f"{self.name}:{query.query_id}",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            schema=POINTWISE_SCHEMA,
            metadata={"query_id": query.query_id, "doc_ids": [doc.doc_id for doc in documents]},
        )
        parsed_results = response.get("results", response) if isinstance(response, dict) else response
        grades = {document.doc_id: 0 for document in documents}
        if isinstance(parsed_results, list):
            for item in parsed_results:
                if not isinstance(item, dict):
                    logger.debug(
                        "Ignoring non-object pointwise result item: query_id=%s, item=%r",
                        query.query_id,
                        item,
                    )
                    continue
                doc_id = item.get("doc_id")
                grade = item.get("grade")
                if doc_id in grades and isinstance(grade, int):
                    grades[doc_id] = max(0, min(3, grade))
                else:
                    logger.debug(
                        "Ignoring invalid pointwise result item: query_id=%s, item=%r",
                        query.query_id,
                        item,
                    )
        else:
            logger.warning(
                "Pointwise provider response is not a list: query_id=%s, response_type=%s",
                query.query_id,
                type(parsed_results).__name__,
            )
        ordered = sorted(
            documents,
            key=lambda document: (-grades[document.doc_id], document.original_rank),
        )
        self._record(
            queries=1,
            provider_calls=1,
            documents_considered=len(documents),
        )
        logger.debug(
            "Pointwise rerank finished: query_id=%s, ordered_doc_ids=%s, grades=%s",
            query.query_id,
            [document.doc_id for document in ordered],
            grades,
        )
        return RerankResult(
            ordered_doc_ids=[document.doc_id for document in ordered],
            metadata={"grades": grades},
        )

    def _build_prompts(self, *, query_text: str, document_payload: str) -> tuple[str, str]:
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
            document_payload=document_payload,
        )
        return system_prompt, user_prompt
