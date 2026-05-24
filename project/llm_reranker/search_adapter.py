from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from mteb.models import ModelMeta

from .strategies import BaseRerankStrategy
from .types import CandidateDocument, QueryExample
from .utils import combine_title_and_text, make_descending_scores


logger = logging.getLogger(__name__)


class LLMRerankerSearchModel:
    """MTEB SearchProtocol adapter for group-wise LLM reranking."""

    def __init__(
        self,
        *,
        provider_name: str,
        provider_model_name: str,
        strategy: BaseRerankStrategy,
        rerank_top_k: int,
        concurrency: int,
    ) -> None:
        self.provider_name = provider_name
        self.provider_model_name = provider_model_name
        self.strategy = strategy
        self.rerank_top_k = rerank_top_k
        self.concurrency = concurrency
        self.task_corpus = None
        self._lock = threading.Lock()
        self._stats = {
            "queries_processed": 0,
            "queries_reranked": 0,
            "documents_reranked": 0,
            "candidates_total": 0,
        }
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites={
                "name": f"{provider_name}:{provider_model_name}:{strategy.name}",
                "framework": ["API"],
                "model_type": ["router"],
                "open_weights": provider_name == "ollama",
            }
        )
        logger.info(
            "LLM reranker search model initialized: provider=%s, model=%s, "
            "strategy=%s, rerank_top_k=%s, concurrency=%s",
            provider_name,
            provider_model_name,
            strategy.name,
            rerank_top_k,
            concurrency,
        )

    def index(
        self,
        corpus,
        *,
        task_metadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs,
        num_proc: int | None = None,
    ) -> None:
        self.task_corpus = corpus
        logger.info(
            "Indexed corpus for search: task=%s, subset=%s, split=%s, documents=%s",
            getattr(task_metadata, "name", task_metadata),
            hf_subset,
            hf_split,
            len(corpus),
        )

    def search(
        self,
        queries,
        *,
        task_metadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs,
        top_ranked=None,
        num_proc: int | None = None,
    ) -> dict[str, dict[str, float]]:
        if top_ranked is None:
            logger.error("Search called without top_ranked candidates")
            raise ValueError("LLMRerankerSearchModel requires top_ranked candidates")
        if self.task_corpus is None:
            logger.error("Search called before corpus indexing")
            raise ValueError("Corpus must be indexed before search")

        doc_lookup = {row["id"]: row for row in self.task_corpus}
        query_lookup = {row["id"]: row for row in queries}
        all_query_ids: list[str] = list(queries["id"])
        results: dict[str, dict[str, float]] = {query_id: {} for query_id in all_query_ids}
        query_ids: list[str] = [query_id for query_id in all_query_ids if query_id in top_ranked]
        logger.info(
            "Starting search rerank: task=%s, subset=%s, split=%s, queries=%s, "
            "with_top_ranked=%s, corpus_docs=%s, top_k=%s, rerank_top_k=%s, "
            "concurrency=%s",
            getattr(task_metadata, "name", task_metadata),
            hf_subset,
            hf_split,
            len(all_query_ids),
            len(query_ids),
            len(doc_lookup),
            top_k,
            self.rerank_top_k,
            self.concurrency,
        )

        def rerank_single_query(query_id: str) -> tuple[str, dict[str, float]]:
            ranked_doc_ids = list(top_ranked.get(query_id, []))
            if not ranked_doc_ids:
                self._record(queries_processed=1)
                logger.debug("Query has no top_ranked candidates: query_id=%s", query_id)
                return query_id, {}

            query_row = query_lookup[query_id]
            query_text = self._query_text(query_row)
            prefix_ids = ranked_doc_ids[: self.rerank_top_k]
            tail_ids = ranked_doc_ids[self.rerank_top_k :]
            missing_prefix_ids = [doc_id for doc_id in prefix_ids if doc_id not in doc_lookup]
            if missing_prefix_ids:
                logger.warning(
                    "Skipping missing candidate documents: query_id=%s, missing=%s",
                    query_id,
                    missing_prefix_ids,
                )
            prefix_docs = [
                CandidateDocument(
                    doc_id=doc_id,
                    text=self._document_text(doc_lookup[doc_id]),
                    original_rank=index,
                )
                for index, doc_id in enumerate(prefix_ids)
                if doc_id in doc_lookup
            ]
            if len(prefix_docs) <= 1:
                logger.debug(
                    "Query skipped strategy rerank because prefix has %s document(s): "
                    "query_id=%s, candidates=%s",
                    len(prefix_docs),
                    query_id,
                    len(ranked_doc_ids),
                )
                final_ids = [doc.doc_id for doc in prefix_docs] + tail_ids
            else:
                logger.debug(
                    "Reranking query prefix: query_id=%s, candidates=%s, prefix=%s, tail=%s",
                    query_id,
                    len(ranked_doc_ids),
                    len(prefix_docs),
                    len(tail_ids),
                )
                reranked = self.strategy.rerank(
                    QueryExample(query_id=query_id, text=query_text),
                    prefix_docs,
                )
                final_ids = reranked.ordered_doc_ids + tail_ids
                self._record(
                    queries_reranked=1,
                    documents_reranked=len(prefix_docs),
                )

            self._record(
                queries_processed=1,
                candidates_total=len(ranked_doc_ids),
            )
            logger.debug(
                "Query rerank completed: query_id=%s, returned_docs=%s",
                query_id,
                len(final_ids),
            )
            return query_id, make_descending_scores(final_ids)

        if self.concurrency <= 1 or len(query_ids) <= 1:
            for query_id in query_ids:
                result_query_id, query_scores = rerank_single_query(query_id)
                results[result_query_id] = query_scores
        else:
            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                for result_query_id, query_scores in executor.map(rerank_single_query, query_ids):
                    results[result_query_id] = query_scores

        self.task_corpus = None
        logger.info("Search rerank finished: stats=%s", self.stats())
        return results

    def stats(self) -> dict[str, Any]:
        with self._lock:
            payload = dict(self._stats)
        payload.update(
            {
                "provider": self.provider_name,
                "model": self.provider_model_name,
                "strategy": self.strategy.name,
                "rerank_top_k": self.rerank_top_k,
                "concurrency": self.concurrency,
            }
        )
        return payload

    def _query_text(self, query_row: dict[str, Any]) -> str:
        instruction = str(query_row.get("instruction", "") or "").strip()
        text = str(query_row.get("text", "") or "").strip()
        if instruction:
            return f"{instruction}\n\n{text}".strip()
        return text

    def _document_text(self, document_row: dict[str, Any]) -> str:
        return combine_title_and_text(
            str(document_row.get("title", "") or ""),
            str(document_row.get("text", "") or ""),
        )

    def _record(self, **updates: int) -> None:
        with self._lock:
            for key, value in updates.items():
                self._stats[key] = self._stats.get(key, 0) + value
