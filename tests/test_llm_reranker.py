from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from datasets import Dataset

from project.llm_reranker.cache import RequestCache
from project.llm_reranker.providers import VLLMProvider
from project.llm_reranker.providers.base import BaseLLMProvider
from project.llm_reranker.search_adapter import LLMRerankerSearchModel
from project.llm_reranker.strategies.listwise import ListwiseRankGPTRerankStrategy
from project.llm_reranker.strategies.pairwise import PairwisePRPRerankStrategy
from project.llm_reranker.strategies.pointwise import PointwiseGradedRerankStrategy
from project.llm_reranker.types import CandidateDocument, QueryExample, RerankResult
from project.run_rumteb_llm_reranker import create_provider, main as llm_runner_main


class QueueProvider(BaseLLMProvider):
    provider_name = "fake"

    def __init__(self, responses, request_cache=None, max_retries=3):
        super().__init__(
            model="fake-model",
            request_cache=request_cache,
            temperature=0.0,
            timeout=1.0,
            max_retries=max_retries,
        )
        self.responses = list(responses)
        self.network_calls = 0

    def _generate_content(self, *, messages, schema):
        del messages, schema
        self.network_calls += 1
        if not self.responses:
            raise RuntimeError("no fake responses left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response, {"provider": "fake"}


class ReversingStrategy:
    name = "reverse"

    def __init__(self):
        self.calls = 0

    def rerank(self, query, documents):
        del query
        self.calls += 1
        return RerankResult(ordered_doc_ids=[document.doc_id for document in reversed(documents)])

    def stats(self):
        return {"strategy": self.name, "calls": self.calls}


class SerializableTaskResult:
    def __init__(self, task_name: str):
        self.task_name = task_name

    def model_dump(self, mode="json"):
        del mode
        return {
            "task_name": self.task_name,
            "scores": {
                "test": [
                    {
                        "ndcg_at_10": 0.5,
                        "map_at_1000": 0.4,
                        "main_score": 0.5,
                    }
                ]
            },
        }


class PointwiseStrategyTests(unittest.TestCase):
    def test_pointwise_reranks_and_preserves_ties_by_original_rank(self):
        provider = QueueProvider(
            [
                {
                    "results": [
                        {"doc_id": "d1", "grade": 1},
                        {"doc_id": "d2", "grade": 3},
                        {"doc_id": "d3", "grade": 3},
                    ]
                }
            ]
        )
        strategy = PointwiseGradedRerankStrategy(
            provider=provider,
            prompt_language="ru",
            query_max_chars=500,
            doc_max_chars=1400,
        )
        docs = [
            CandidateDocument("d1", "doc1", 0),
            CandidateDocument("d2", "doc2", 1),
            CandidateDocument("d3", "doc3", 2),
        ]
        result = strategy.rerank(QueryExample("q1", "query"), docs)
        self.assertEqual(result.ordered_doc_ids, ["d2", "d3", "d1"])
        self.assertEqual(provider.network_calls, 1)


class PairwiseStrategyTests(unittest.TestCase):
    def test_pairwise_uses_request_cache_between_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = RequestCache(Path(temp_dir))
            provider = QueueProvider(
                [
                    {"winner_doc_id": "d2"},
                    {"winner_doc_id": "d2"},
                    {"winner_doc_id": "d3"},
                ],
                request_cache=cache,
            )
            strategy = PairwisePRPRerankStrategy(
                provider=provider,
                prompt_language="ru",
                query_max_chars=500,
                doc_max_chars=900,
            )
            docs = [
                CandidateDocument("d1", "doc1", 0),
                CandidateDocument("d2", "doc2", 1),
                CandidateDocument("d3", "doc3", 2),
            ]
            first = strategy.rerank(QueryExample("q1", "query"), docs)
            second = strategy.rerank(QueryExample("q1", "query"), docs)

            self.assertEqual(first.ordered_doc_ids, ["d2", "d3", "d1"])
            self.assertEqual(second.ordered_doc_ids, ["d2", "d3", "d1"])
            self.assertEqual(provider.network_calls, 3)
            self.assertGreater(cache.stats()["hits"], 0)


class ListwiseStrategyTests(unittest.TestCase):
    def test_listwise_applies_sliding_windows(self):
        provider = QueueProvider(
            [
                {"ordered_doc_ids": ["d5", "d6", "d3", "d4"]},
                {"ordered_doc_ids": ["d5", "d1", "d6", "d2"]},
            ]
        )
        strategy = ListwiseRankGPTRerankStrategy(
            provider=provider,
            prompt_language="ru",
            query_max_chars=500,
            doc_max_chars=500,
            window_size=4,
            stride=2,
        )
        docs = [
            CandidateDocument("d1", "doc1", 0),
            CandidateDocument("d2", "doc2", 1),
            CandidateDocument("d3", "doc3", 2),
            CandidateDocument("d4", "doc4", 3),
            CandidateDocument("d5", "doc5", 4),
            CandidateDocument("d6", "doc6", 5),
        ]
        result = strategy.rerank(QueryExample("q1", "query"), docs)
        self.assertEqual(result.ordered_doc_ids, ["d5", "d1", "d6", "d2", "d3", "d4"])
        self.assertEqual(provider.network_calls, 2)


class SearchAdapterTests(unittest.TestCase):
    def test_search_adapter_reranks_only_prefix_and_keeps_tail(self):
        strategy = ReversingStrategy()
        model = LLMRerankerSearchModel(
            provider_name="fake",
            provider_model_name="fake-model",
            strategy=strategy,
            rerank_top_k=3,
            concurrency=1,
        )
        corpus = Dataset.from_list(
            [
                {"id": "d1", "title": "", "text": "doc1"},
                {"id": "d2", "title": "", "text": "doc2"},
                {"id": "d3", "title": "", "text": "doc3"},
                {"id": "d4", "title": "", "text": "doc4"},
                {"id": "d5", "title": "", "text": "doc5"},
            ]
        )
        queries = Dataset.from_list([{"id": "q1", "text": "query"}])
        model.index(
            corpus,
            task_metadata=None,
            hf_split="test",
            hf_subset="default",
            encode_kwargs={},
            num_proc=None,
        )
        results = model.search(
            queries,
            task_metadata=None,
            hf_split="test",
            hf_subset="default",
            top_k=1000,
            encode_kwargs={},
            top_ranked={"q1": ["d1", "d2", "d3", "d4", "d5"]},
            num_proc=None,
        )
        self.assertEqual(
            list(results["q1"].keys()),
            ["d3", "d2", "d1", "d4", "d5"],
        )
        self.assertEqual(
            list(results["q1"].values()),
            [5.0, 4.0, 3.0, 2.0, 1.0],
        )


class ProviderCacheRetryTests(unittest.TestCase):
    def test_provider_retries_malformed_json_and_then_hits_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = RequestCache(Path(temp_dir))
            provider = QueueProvider(
                ["not valid json", '{"results":[{"doc_id":"d1","grade":2}]}'],
                request_cache=cache,
                max_retries=2,
            )
            parsed = provider.generate_json(
                cache_namespace="ns",
                messages=[{"role": "user", "content": "hello"}],
            )
            cached = provider.generate_json(
                cache_namespace="ns",
                messages=[{"role": "user", "content": "hello"}],
            )

            self.assertEqual(parsed["results"][0]["doc_id"], "d1")
            self.assertEqual(cached["results"][0]["grade"], 2)
            self.assertEqual(provider.network_calls, 2)
            self.assertEqual(cache.stats()["hits"], 1)


class VLLMProviderTests(unittest.TestCase):
    def test_vllm_sends_openai_compatible_chat_completion_request(self):
        provider = VLLMProvider(
            model="Qwen/Qwen3-4B-Instruct-2507",
            request_cache=None,
            temperature=0.0,
            timeout=5.0,
            max_retries=1,
            base_url="http://127.0.0.1:8000/v1/",
            api_key="token-abc123",
        )
        response = mock.Mock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": '{"ok": true}'}}],
        }
        response.raise_for_status = mock.Mock()

        with mock.patch(
            "project.llm_reranker.providers.vllm.requests.post",
            return_value=response,
        ) as post:
            parsed = provider.generate_json(
                cache_namespace="ns",
                messages=[{"role": "user", "content": "hello"}],
                schema={
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                },
            )

        self.assertEqual(parsed, {"ok": True})
        post.assert_called_once()
        self.assertEqual(
            post.call_args.args[0],
            "http://127.0.0.1:8000/v1/chat/completions",
        )
        self.assertEqual(post.call_args.kwargs["timeout"], 5.0)
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer token-abc123",
        )
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "Qwen/Qwen3-4B-Instruct-2507")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])

    def test_runner_creates_vllm_provider_from_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            request_cache = RequestCache(Path(temp_dir))
            args = SimpleNamespace(
                provider="vllm",
                model="served-model",
                base_url=None,
                api_key=None,
                temperature=0.1,
                timeout=10.0,
                max_retries=2,
            )
            with mock.patch.dict(
                "os.environ",
                {
                    "VLLM_BASE_URL": "http://localhost:9000/v1",
                    "VLLM_API_KEY": "local-token",
                },
            ):
                provider = create_provider(args, request_cache)

        self.assertIsInstance(provider, VLLMProvider)
        self.assertEqual(provider.base_url, "http://localhost:9000/v1")
        self.assertEqual(provider.api_key, "local-token")


class RunnerSmokeTests(unittest.TestCase):
    def test_cli_smoke_writes_summary_and_run_stats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            fake_provider = QueueProvider([])
            fake_strategy = PointwiseGradedRerankStrategy(
                provider=fake_provider,
                prompt_language="ru",
                query_max_chars=500,
                doc_max_chars=1400,
            )
            fake_tasks = [SimpleNamespace(metadata=SimpleNamespace(name="RuBQReranking"))]
            fake_results = SimpleNamespace(
                task_results=[SerializableTaskResult("RuBQReranking")]
            )
            with (
                mock.patch("project.run_rumteb_llm_reranker.load_tasks", return_value=fake_tasks),
                mock.patch("project.run_rumteb_llm_reranker.subset_task_queries"),
                mock.patch("project.run_rumteb_llm_reranker.create_provider", return_value=fake_provider),
                mock.patch("project.run_rumteb_llm_reranker.create_strategy", return_value=fake_strategy),
                mock.patch("project.run_rumteb_llm_reranker.mteb.evaluate", return_value=fake_results),
            ):
                exit_code = llm_runner_main(
                    [
                        "--provider",
                        "ollama",
                        "--model",
                        "fake-model",
                        "--strategy",
                        "pointwise-graded",
                        "--profile",
                        "quick",
                        "--max-queries-per-task",
                        "2",
                        "--output-dir",
                        str(output_dir),
                        "--skip-preflight",
                        "--quiet",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "run_stats.json").exists())


if __name__ == "__main__":
    unittest.main()
