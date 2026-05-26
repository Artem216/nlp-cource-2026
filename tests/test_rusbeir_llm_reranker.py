from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from datasets import Dataset

from project.llm_reranker.rusbeir import (
    RusBeirDatasetSpec,
    evaluate_retrieval,
    load_rusbeir_queries_and_qrels,
    load_first_stage_run,
    score_entry_from_metrics,
)
from project.llm_reranker.types import RerankResult
from project.run_rusbeir_llm_reranker import (
    main as rusbeir_runner_main,
    parse_args as parse_rusbeir_llm_args,
)


class ReversingStrategy:
    name = "reverse"

    def __init__(self):
        self.calls = 0

    def rerank(self, query, documents):
        del query
        self.calls += 1
        return RerankResult(
            ordered_doc_ids=[document.doc_id for document in reversed(documents)]
        )

    def stats(self):
        return {"strategy": self.name, "calls": self.calls}


class RusBeirRunParserTests(unittest.TestCase):
    def test_load_first_stage_run_supports_trec_and_score_tsv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_path = Path(temp_dir) / "mixed.run"
            run_path.write_text(
                "\n".join(
                    [
                        "q1 Q0 d1 2 4.0 run",
                        "q1 Q0 d2 1 9.0 run",
                        "q2 d3 0.7",
                        "q2 d4 0.9",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            top_ranked = load_first_stage_run(run_path, top_k=2)

        self.assertEqual(top_ranked["q1"], ["d2", "d1"])
        self.assertEqual(top_ranked["q2"], ["d4", "d3"])

    def test_load_first_stage_run_supports_json_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_path = Path(temp_dir) / "run.json"
            run_path.write_text(
                json.dumps({"q1": {"d1": 0.1, "d2": 0.9}, "q2": ["d3", "d4"]}),
                encoding="utf-8",
            )

            top_ranked = load_first_stage_run(run_path)

        self.assertEqual(top_ranked["q1"], ["d2", "d1"])
        self.assertEqual(top_ranked["q2"], ["d3", "d4"])


class RusBeirMetricsTests(unittest.TestCase):
    def test_evaluate_retrieval_computes_beir_style_cut_metrics(self):
        qrels = {
            "q1": {"d1": 1, "d2": 1},
            "q2": {"d3": 2},
        }
        results = {
            "q1": {"d2": 3.0, "x": 2.0, "d1": 1.0},
            "q2": {"d4": 2.0, "d3": 1.0},
        }

        metrics = evaluate_retrieval(qrels, results, k_values=[1, 2])
        score_entry = score_entry_from_metrics(metrics)

        self.assertAlmostEqual(metrics["precision"][1], 0.5)
        self.assertAlmostEqual(metrics["recall"][2], 0.75)
        self.assertAlmostEqual(metrics["mrr"][2], 0.75)
        self.assertIn("ndcg_at_1", score_entry)
        self.assertIsNone(score_entry["main_score"])


class RusBeirLoaderTests(unittest.TestCase):
    def test_load_queries_and_qrels_falls_back_to_train_physical_split(self):
        spec = RusBeirDatasetSpec(
            name="fake-rusbeir",
            corpus_repo="fake-corpus",
            qrels_repo="fake-qrels",
            default_split="test",
            display_name="Fake rusBEIR",
        )
        raw_queries = Dataset.from_list([{"_id": "q1", "text": "query"}])
        raw_qrels = Dataset.from_list(
            [{"query-id": "q1", "corpus-id": "d1", "score": 1}]
        )
        calls = []

        def fake_load_dataset(repo, *configs, split):
            config = configs[0] if configs else None
            calls.append((repo, config, split))
            if split == "test":
                raise ValueError('Unknown split "test". Should be one of [\'train\'].')
            if repo == "fake-corpus":
                self.assertEqual(config, "queries")
                return raw_queries
            if repo == "fake-qrels":
                self.assertIsNone(config)
                return raw_qrels
            raise AssertionError(repo)

        with mock.patch(
            "project.llm_reranker.rusbeir.load_dataset",
            side_effect=fake_load_dataset,
        ):
            queries, qrels = load_rusbeir_queries_and_qrels(
                spec=spec,
                split="test",
                max_queries_per_task=None,
            )

        self.assertEqual(queries["id"], ["q1"])
        self.assertEqual(qrels, {"q1": {"d1": 1}})
        self.assertEqual(
            calls,
            [
                ("fake-corpus", "queries", "test"),
                ("fake-corpus", "queries", "train"),
                ("fake-qrels", None, "test"),
                ("fake-qrels", None, "train"),
            ],
        )

    def test_load_queries_and_qrels_limits_after_qrel_alignment(self):
        spec = RusBeirDatasetSpec(
            name="fake-rusbeir",
            corpus_repo="fake-corpus",
            qrels_repo="fake-qrels",
            default_split="test",
            display_name="Fake rusBEIR",
        )
        raw_queries = Dataset.from_list(
            [
                {"_id": "q0", "text": "unjudged query"},
                {"_id": "q1", "text": "first judged query"},
                {"_id": "q2", "text": "second judged query"},
            ]
        )
        raw_qrels = Dataset.from_list(
            [
                {"query-id": "q1", "corpus-id": "d1", "score": 1},
                {"query-id": "q2", "corpus-id": "d2", "score": 1},
            ]
        )

        def fake_load_dataset(repo, *configs, split):
            self.assertEqual(split, "test")
            if repo == "fake-corpus":
                self.assertEqual(configs, ("queries",))
                return raw_queries
            if repo == "fake-qrels":
                self.assertEqual(configs, ())
                return raw_qrels
            raise AssertionError(repo)

        with mock.patch(
            "project.llm_reranker.rusbeir.load_dataset",
            side_effect=fake_load_dataset,
        ):
            queries, qrels = load_rusbeir_queries_and_qrels(
                spec=spec,
                split="test",
                max_queries_per_task=1,
            )

        self.assertEqual(queries["id"], ["q1"])
        self.assertEqual(qrels, {"q1": {"d1": 1}})


class RusBeirCliTests(unittest.TestCase):
    def test_candidate_source_auto_uses_tfidf_without_first_stage_run(self):
        args = parse_rusbeir_llm_args(
            [
                "--provider",
                "vllm",
                "--model",
                "fake-model",
                "--strategy",
                "pointwise-graded",
            ]
        )

        self.assertEqual(args.candidate_source, "tfidf")

    def test_candidate_source_auto_uses_run_with_first_stage_run(self):
        args = parse_rusbeir_llm_args(
            [
                "--provider",
                "vllm",
                "--model",
                "fake-model",
                "--strategy",
                "pointwise-graded",
                "--first-stage-run",
                "runs/rus-scifact.run",
            ]
        )

        self.assertEqual(args.candidate_source, "run")


class RusBeirRunnerSmokeTests(unittest.TestCase):
    def test_cli_smoke_writes_summary_and_reranked_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "out"
            run_path = root / "rus-scifact.run"
            run_path.write_text(
                "\n".join(
                    [
                        "q1 Q0 d1 1 3.0 first",
                        "q1 Q0 d2 2 2.0 first",
                        "q1 Q0 d3 3 1.0 first",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            queries = Dataset.from_list([{"id": "q1", "text": "query"}])
            qrels = {"q1": {"d2": 1}}
            corpus = Dataset.from_list(
                [
                    {"id": "d1", "title": "", "text": "doc1"},
                    {"id": "d2", "title": "", "text": "doc2"},
                ]
            )
            fake_provider = SimpleNamespace(stats=lambda: {"provider": "fake"})
            fake_strategy = ReversingStrategy()

            def fake_load_corpus(**kwargs):
                self.assertEqual(kwargs["needed_doc_ids"], {"d1", "d2"})
                return corpus

            with (
                mock.patch(
                    "project.run_rusbeir_llm_reranker.load_rusbeir_queries_and_qrels",
                    return_value=(queries, qrels),
                ),
                mock.patch(
                    "project.run_rusbeir_llm_reranker.load_rusbeir_corpus",
                    side_effect=fake_load_corpus,
                ),
                mock.patch(
                    "project.run_rusbeir_llm_reranker.create_provider",
                    return_value=fake_provider,
                ),
                mock.patch(
                    "project.run_rusbeir_llm_reranker.create_strategy",
                    return_value=fake_strategy,
                ),
            ):
                exit_code = rusbeir_runner_main(
                    [
                        "--provider",
                        "ollama",
                        "--model",
                        "fake-model",
                        "--strategy",
                        "pointwise-graded",
                        "--tasks",
                        "rus-scifact",
                        "--first-stage-run",
                        str(run_path),
                        "--rerank-top-k",
                        "2",
                        "--max-queries-per-task",
                        "1",
                        "--output-dir",
                        str(output_dir),
                        "--skip-preflight",
                        "--quiet",
                    ]
                )

            self.assertEqual(exit_code, 0)
            summary_path = output_dir / "summary.json"
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["benchmark"], "rusBEIR")
            self.assertEqual(
                summary["task_results"]["rus-scifact"]["scores"]["test"][0]["ndcg_at_10"],
                1.0,
            )
            run_lines = (output_dir / "predictions" / "rus-scifact.run").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertIn("q1 Q0 d2 1", run_lines[0])
            self.assertEqual(fake_strategy.calls, 1)


if __name__ == "__main__":
    unittest.main()
