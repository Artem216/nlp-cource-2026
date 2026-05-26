from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from datasets import Dataset

from project.simple_llm_reranker.run import (
    GradesCache,
    JsonlEventLogger,
    OpenAICompatibleGradeClient,
    SimplePointwiseDocReranker,
    main as simple_runner_main,
    parse_grade,
)


class FakeResponse:
    def __init__(self, content: str | None = None, status_code: int = 200):
        self.content = content
        self.status_code = status_code
        self.text = content or ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


class QueueSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.posts = []
        self.gets = []

    def post(self, *args, **kwargs):
        self.posts.append((args, kwargs))
        if not self.responses:
            raise RuntimeError("no fake responses left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def get(self, *args, **kwargs):
        self.gets.append((args, kwargs))
        return FakeResponse('{"ok": true}')


def build_model(temp_dir: str, responses, *, max_retries: int = 2):
    event_logger = JsonlEventLogger(Path(temp_dir) / "events.jsonl")
    session = QueueSession(responses)
    client = OpenAICompatibleGradeClient(
        base_url="https://example.test/v1/",
        model="served-model",
        api_key="token-abc",
        timeout=5.0,
        max_retries=max_retries,
        max_tokens=32,
        temperature=0.0,
        event_logger=event_logger,
        session=session,
        sleep_fn=lambda _: None,
    )
    model = SimplePointwiseDocReranker(
        client=client,
        grades_cache=GradesCache(Path(temp_dir) / "grades_cache"),
        event_logger=event_logger,
        rerank_top_k=3,
        query_max_chars=300,
        doc_max_chars=700,
    )
    return model, session


class SimpleGradeClientTests(unittest.TestCase):
    def test_client_sends_openai_compatible_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            event_logger = JsonlEventLogger(Path(temp_dir) / "events.jsonl")
            session = QueueSession([FakeResponse('{"grade": 2}')])
            client = OpenAICompatibleGradeClient(
                base_url="https://api.example/v1/",
                model="served-model",
                api_key="secret-token",
                timeout=7.0,
                max_retries=1,
                max_tokens=32,
                temperature=0.0,
                event_logger=event_logger,
                session=session,
                sleep_fn=lambda _: None,
            )

            grade, _ = client.grade_document(
                query_id="q1",
                doc_id="d1",
                query_text="query",
                doc_text="doc",
                query_index=1,
                query_total=1,
                doc_index=1,
                doc_total=1,
            )

            self.assertEqual(grade, 2)
            self.assertEqual(len(session.posts), 1)
            args, kwargs = session.posts[0]
            self.assertEqual(args[0], "https://api.example/v1/chat/completions")
            self.assertEqual(kwargs["timeout"], 7.0)
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret-token")
            payload = kwargs["json"]
            self.assertEqual(payload["model"], "served-model")
            self.assertEqual(payload["temperature"], 0.0)
            self.assertEqual(payload["max_tokens"], 32)
            self.assertNotIn("response_format", payload)

    def test_parse_grade_uses_json_then_regex_fallback(self):
        self.assertEqual(parse_grade('{"grade": 3}'), 3)
        self.assertEqual(parse_grade('```json\n{"grade": 1}\n```'), 1)

    def test_malformed_response_retries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model, session = build_model(
                temp_dir,
                [FakeResponse("not json"), FakeResponse('{"grade": 2}')],
                max_retries=2,
            )
            grade, _ = model.client.grade_document(
                query_id="q1",
                doc_id="d1",
                query_text="query",
                doc_text="doc",
                query_index=1,
                query_total=1,
                doc_index=1,
                doc_total=1,
            )

            self.assertEqual(grade, 2)
            self.assertEqual(len(session.posts), 2)
            self.assertEqual(model.client.stats()["provider_retries"], 1)


class SimplePointwiseDocRerankerTests(unittest.TestCase):
    def test_search_grades_one_doc_per_request_and_preserves_tail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model, session = build_model(
                temp_dir,
                [
                    FakeResponse('{"grade": 1}'),
                    FakeResponse('{"grade": 3}'),
                    FakeResponse('{"grade": 2}'),
                ],
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
                task_metadata=SimpleNamespace(name="FakeTask"),
                hf_split="test",
                hf_subset="default",
                encode_kwargs={},
            )
            results = model.search(
                queries,
                task_metadata=SimpleNamespace(name="FakeTask"),
                hf_split="test",
                hf_subset="default",
                top_k=1000,
                encode_kwargs={},
                top_ranked={"q1": ["d1", "d2", "d3", "d4", "d5"]},
            )

            self.assertEqual(list(results["q1"].keys()), ["d2", "d3", "d1", "d4", "d5"])
            self.assertEqual(len(session.posts), 3)
            sent_prompts = [call[1]["json"]["messages"][1]["content"] for call in session.posts]
            self.assertIn("Document doc_id: d1", sent_prompts[0])
            self.assertIn("Document doc_id: d2", sent_prompts[1])
            self.assertIn("Document doc_id: d3", sent_prompts[2])

    def test_permanent_doc_failure_returns_original_query_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model, _ = build_model(temp_dir, [FakeResponse("bad")], max_retries=1)
            corpus = Dataset.from_list(
                [
                    {"id": "d1", "title": "", "text": "doc1"},
                    {"id": "d2", "title": "", "text": "doc2"},
                    {"id": "d3", "title": "", "text": "doc3"},
                ]
            )
            queries = Dataset.from_list([{"id": "q1", "text": "query"}])
            model.index(
                corpus,
                task_metadata=None,
                hf_split="test",
                hf_subset="default",
                encode_kwargs={},
            )
            results = model.search(
                queries,
                task_metadata=None,
                hf_split="test",
                hf_subset="default",
                top_k=1000,
                encode_kwargs={},
                top_ranked={"q1": ["d1", "d2", "d3"]},
            )

            self.assertEqual(list(results["q1"].keys()), ["d1", "d2", "d3"])
            self.assertEqual(model.stats()["queries_skipped_due_to_doc_failure"], 1)

    def test_cache_hit_skips_http_request_with_rerankable_prefix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model, session = build_model(
                temp_dir,
                [FakeResponse('{"grade": 3}'), FakeResponse('{"grade": 1}')],
            )
            corpus = Dataset.from_list(
                [
                    {"id": "d1", "title": "", "text": "doc1"},
                    {"id": "d2", "title": "", "text": "doc2"},
                ]
            )
            queries = Dataset.from_list([{"id": "q1", "text": "query"}])
            model.index(corpus, task_metadata=None, hf_split="test", hf_subset="default", encode_kwargs={})
            model.search(
                queries,
                task_metadata=None,
                hf_split="test",
                hf_subset="default",
                top_k=1000,
                encode_kwargs={},
                top_ranked={"q1": ["d1", "d2"]},
            )
            model.index(corpus, task_metadata=None, hf_split="test", hf_subset="default", encode_kwargs={})
            model.search(
                queries,
                task_metadata=None,
                hf_split="test",
                hf_subset="default",
                top_k=1000,
                encode_kwargs={},
                top_ranked={"q1": ["d1", "d2"]},
            )

            self.assertEqual(len(session.posts), 2)
            self.assertEqual(model.stats()["documents_from_cache"], 2)


class SimpleRunnerSmokeTests(unittest.TestCase):
    def test_main_writes_summary_and_run_stats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            fake_tasks = [SimpleNamespace(metadata=SimpleNamespace(name="RuBQReranking"))]
            fake_results = SimpleNamespace(
                task_results=[
                    SimpleNamespace(
                        model_dump=lambda mode="json": {
                            "task_name": "RuBQReranking",
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
                    )
                ]
            )
            with (
                mock.patch("project.simple_llm_reranker.run.load_tasks", return_value=fake_tasks),
                mock.patch("project.simple_llm_reranker.run.subset_task_queries"),
                mock.patch("project.simple_llm_reranker.run.mteb.evaluate", return_value=fake_results),
            ):
                exit_code = simple_runner_main(
                    [
                        "--model",
                        "served-model",
                        "--output-dir",
                        str(output_dir),
                        "--skip-preflight",
                        "--tasks",
                        "RuBQReranking",
                        "--quiet",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "run.log").exists())
            self.assertTrue((output_dir / "events.jsonl").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "run_stats.json").exists())


if __name__ == "__main__":
    unittest.main()
