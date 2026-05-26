from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from datasets import Dataset

from project.run_rusbeir_reranker import (
    main as rusbeir_cross_encoder_main,
    parse_args as parse_rusbeir_cross_encoder_args,
    rerank_with_cross_encoder,
)


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = []
        self.max_seq_length = 512

    def predict(self, pairs, **kwargs):
        self.calls.append((list(pairs), dict(kwargs)))
        count = len(pairs)
        if len(self.scores) < count:
            raise RuntimeError("not enough fake scores")
        scores = self.scores[:count]
        self.scores = self.scores[count:]
        return scores


class RusBeirCrossEncoderTests(unittest.TestCase):
    def test_candidate_source_auto_uses_tfidf_without_first_stage_run(self):
        args = parse_rusbeir_cross_encoder_args(["fake-cross-encoder"])

        self.assertEqual(args.candidate_source, "tfidf")

    def test_candidate_source_auto_uses_run_with_first_stage_run(self):
        args = parse_rusbeir_cross_encoder_args(
            ["fake-cross-encoder", "--first-stage-run", "runs/rus-scifact.run"]
        )

        self.assertEqual(args.candidate_source, "run")

    def test_rerank_with_cross_encoder_scores_prefix_and_preserves_tail(self):
        model = FakeCrossEncoder([0.1, 0.9, 0.2])
        corpus = Dataset.from_list(
            [
                {"id": "d1", "title": "", "text": "doc1"},
                {"id": "d2", "title": "", "text": "doc2"},
                {"id": "d3", "title": "", "text": "doc3"},
                {"id": "d4", "title": "", "text": "doc4"},
            ]
        )
        queries = Dataset.from_list([{"id": "q1", "text": "query"}])

        results, stats = rerank_with_cross_encoder(
            model=model,
            corpus=corpus,
            queries=queries,
            top_ranked={"q1": ["d1", "d2", "d3", "d4"]},
            rerank_top_k=3,
            batch_size=8,
            show_progress_bar=False,
        )

        self.assertEqual(list(results["q1"].keys()), ["d2", "d3", "d1", "d4"])
        self.assertEqual(stats["documents_scored"], 3)
        self.assertEqual(model.calls[0][1]["batch_size"], 8)
        self.assertFalse(model.calls[0][1]["show_progress_bar"])

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
            fake_model = FakeCrossEncoder([0.1, 0.9])

            def fake_load_corpus(**kwargs):
                self.assertEqual(kwargs["needed_doc_ids"], {"d1", "d2"})
                return corpus

            with (
                mock.patch(
                    "project.run_rusbeir_reranker.load_rusbeir_queries_and_qrels",
                    return_value=(queries, qrels),
                ),
                mock.patch(
                    "project.run_rusbeir_reranker.load_rusbeir_corpus",
                    side_effect=fake_load_corpus,
                ),
                mock.patch(
                    "project.run_rusbeir_reranker.CrossEncoder",
                    return_value=fake_model,
                ),
            ):
                exit_code = rusbeir_cross_encoder_main(
                    [
                        "fake-cross-encoder",
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
                        "--device",
                        "cpu",
                        "--quiet",
                    ]
                )

            self.assertEqual(exit_code, 0)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["benchmark"], "rusBEIR")
            self.assertEqual(summary["model_name_or_path"], "fake-cross-encoder")
            self.assertEqual(
                summary["task_results"]["rus-scifact"]["scores"]["test"][0]["ndcg_at_10"],
                1.0,
            )
            run_lines = (output_dir / "predictions" / "rus-scifact.run").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertIn("q1 Q0 d2 1", run_lines[0])


if __name__ == "__main__":
    unittest.main()
