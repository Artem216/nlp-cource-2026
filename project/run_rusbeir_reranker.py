#!/usr/bin/env python3
"""Run a cross-encoder model on rusBEIR datasets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from sentence_transformers import CrossEncoder

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from project.llm_reranker.benchmark import positive_int  # noqa: E402
from project.llm_reranker.logging_utils import add_logging_arguments, configure_logging  # noqa: E402
from project.llm_reranker.rusbeir import (  # noqa: E402
    RUSBEIR_BENCHMARK_NAME,
    RUSBEIR_BENCHMARK_VERSION,
    RUSBEIR_K_VALUES,
    RusBeirDatasetSpec,
    available_rusbeir_task_names,
    build_tfidf_first_stage,
    evaluate_retrieval,
    limit_top_ranked_to_queries,
    load_first_stage_run,
    load_rusbeir_corpus,
    load_rusbeir_queries_and_qrels,
    needed_prefix_doc_ids,
    resolve_first_stage_run_path,
    resolve_rusbeir_dataset_specs,
    score_entry_from_metrics,
    split_for_spec,
    top_ranked_to_scores,
    write_json_results,
    write_trec_run,
)
from project.llm_reranker.utils import (  # noqa: E402
    combine_title_and_text,
    make_descending_scores,
    sanitize_model_name,
)
from project.run_rumteb_reranker import (  # noqa: E402
    device_value,
    model_max_length,
    resolve_device,
)


logger = logging.getLogger(__name__)

CANDIDATE_SOURCES = ("auto", "run", "tfidf")
CROSS_ENCODER_PROFILE_DEFAULTS = {
    "full": {"rerank_top_k": 50, "max_queries_per_task": None},
    "quick": {"rerank_top_k": 100, "max_queries_per_task": 25},
}
SPLITS = ("auto", "train", "dev", "test")
TEXT_TYPES = ("processed_text", "text")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a cross-encoder model on rusBEIR datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_name_or_path", help="Hugging Face model id or local path")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["rus-scifact"],
        help=(
            "rusBEIR datasets to evaluate. Use 'all' for the full set. "
            f"Known names: {', '.join(available_rusbeir_task_names())}"
        ),
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default="auto",
        help="Dataset split. 'auto' uses the rusBEIR default split per dataset.",
    )
    parser.add_argument(
        "--text-type",
        choices=TEXT_TYPES,
        default="processed_text",
        help="Corpus field preferred for prompts and TF-IDF candidates.",
    )
    parser.add_argument(
        "--candidate-source",
        choices=CANDIDATE_SOURCES,
        default="auto",
        help=(
            "Where first-stage candidates come from. "
            "'auto' uses --first-stage-run when provided, otherwise TF-IDF."
        ),
    )
    parser.add_argument(
        "--first-stage-run",
        type=Path,
        default=None,
        help=(
            "TREC/TSV/JSON/JSONL first-stage run file, or a directory containing "
            "<task>.run files. Required when --candidate-source run."
        ),
    )
    parser.add_argument(
        "--first-stage-top-k",
        type=positive_int,
        default=1000,
        help="Number of first-stage candidates kept per query.",
    )
    parser.add_argument(
        "--tfidf-max-features",
        type=positive_int,
        default=200_000,
        help="Maximum vocabulary size for --candidate-source tfidf.",
    )
    parser.add_argument(
        "--max-corpus-docs",
        type=positive_int,
        default=None,
        help="Optional corpus cap, mainly for quick TF-IDF smoke runs.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(CROSS_ENCODER_PROFILE_DEFAULTS),
        default="quick",
        help="Benchmark profile controlling default query and rerank limits.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=positive_int,
        default=None,
        help="Number of top-ranked candidates rescored with the cross-encoder.",
    )
    parser.add_argument(
        "--max-queries-per-task",
        type=positive_int,
        default=None,
        help="Optional query cap per dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for predictions, run stats, and summary.json.",
    )
    parser.add_argument(
        "--device",
        type=device_value,
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=32,
        help="Batch size passed to CrossEncoder.predict.",
    )
    parser.add_argument(
        "--max-length",
        type=positive_int,
        default=None,
        help="Optional max input length passed to CrossEncoder.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision to pass to CrossEncoder.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce progress output.")
    add_logging_arguments(parser)
    args = parser.parse_args(argv)
    if args.candidate_source == "auto":
        args.candidate_source = "run" if args.first_stage_run is not None else "tfidf"
    return args


def resolve_profile_args(args: argparse.Namespace) -> tuple[int, int | None]:
    defaults = CROSS_ENCODER_PROFILE_DEFAULTS[args.profile]
    rerank_top_k = args.rerank_top_k or defaults["rerank_top_k"]
    if args.max_queries_per_task is not None:
        max_queries_per_task = args.max_queries_per_task
    else:
        max_queries_per_task = defaults["max_queries_per_task"]
    logger.info(
        "Resolved profile limits: profile=%s, rerank_top_k=%s, max_queries_per_task=%s",
        args.profile,
        rerank_top_k,
        max_queries_per_task,
    )
    return rerank_top_k, max_queries_per_task


def resolve_output_dir(model_name_or_path: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    script_dir = Path(__file__).resolve().parent
    date_part = datetime.now().date().isoformat()
    return (
        script_dir
        / "results"
        / f"rusbeir_cross_encoder_runs_{date_part}"
        / sanitize_model_name(model_name_or_path)
    )


def candidate_source_for_task(
    *,
    args: argparse.Namespace,
    spec: RusBeirDatasetSpec,
    queries,
    corpus=None,
) -> tuple[dict[str, list[str]], str | None]:
    query_ids = [str(query_id) for query_id in queries["id"]]
    if args.candidate_source == "run":
        if args.first_stage_run is None:
            raise ValueError("--first-stage-run is required when --candidate-source run")
        run_path = resolve_first_stage_run_path(args.first_stage_run, spec)
        top_ranked = load_first_stage_run(run_path, top_k=args.first_stage_top_k)
        return (
            limit_top_ranked_to_queries(top_ranked, query_ids, top_k=args.first_stage_top_k),
            str(run_path.resolve()),
        )

    if corpus is None:
        raise ValueError("TF-IDF candidate source requires loaded corpus")
    top_ranked = build_tfidf_first_stage(
        corpus=corpus,
        queries=queries,
        top_k=args.first_stage_top_k,
        max_features=args.tfidf_max_features,
    )
    return top_ranked, None


def rerank_with_cross_encoder(
    *,
    model: CrossEncoder,
    corpus,
    queries,
    top_ranked: dict[str, list[str]],
    rerank_top_k: int,
    batch_size: int,
    show_progress_bar: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    doc_lookup = {str(row["id"]): row for row in corpus}
    results: dict[str, dict[str, float]] = {}
    stats = {
        "queries_processed": 0,
        "queries_reranked": 0,
        "queries_skipped_due_to_missing_docs": 0,
        "documents_scored": 0,
        "candidates_total": 0,
    }

    for query_row in queries:
        query_id = str(query_row["id"])
        ranked_doc_ids = list(top_ranked.get(query_id, []))
        stats["queries_processed"] += 1
        stats["candidates_total"] += len(ranked_doc_ids)
        if not ranked_doc_ids:
            results[query_id] = {}
            continue

        prefix_ids = ranked_doc_ids[:rerank_top_k]
        tail_ids = ranked_doc_ids[rerank_top_k:]
        missing_prefix_ids = [doc_id for doc_id in prefix_ids if doc_id not in doc_lookup]
        if missing_prefix_ids:
            logger.warning(
                "Query fallback to original order because prefix docs are missing: "
                "query_id=%s, missing=%s",
                query_id,
                missing_prefix_ids,
            )
            stats["queries_skipped_due_to_missing_docs"] += 1
            results[query_id] = make_descending_scores(ranked_doc_ids)
            continue

        query_text = str(query_row.get("text", "") or "")
        pairs = [
            [query_text, _document_text(doc_lookup[doc_id])]
            for doc_id in prefix_ids
        ]
        raw_scores = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        scores = [_score_to_float(score) for score in raw_scores]
        scored_prefix = sorted(
            zip(prefix_ids, scores, range(len(prefix_ids)), strict=True),
            key=lambda item: (-item[1], item[2]),
        )
        final_ids = [doc_id for doc_id, _, _ in scored_prefix] + tail_ids
        results[query_id] = make_descending_scores(final_ids)
        stats["queries_reranked"] += 1
        stats["documents_scored"] += len(prefix_ids)

    return results, stats


def evaluate_results(results: dict[str, dict[str, float]], qrels: dict[str, dict[str, int]]):
    return score_entry_from_metrics(evaluate_retrieval(qrels, results, k_values=RUSBEIR_K_VALUES))


def run_task(
    *,
    args: argparse.Namespace,
    spec: RusBeirDatasetSpec,
    split: str,
    model: CrossEncoder,
    prediction_dir: Path,
    rerank_top_k: int,
    max_queries_per_task: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    logger.info("Starting rusBEIR cross-encoder task: task=%s, split=%s", spec.name, split)
    queries, qrels = load_rusbeir_queries_and_qrels(
        spec=spec,
        split=split,
        max_queries_per_task=max_queries_per_task,
    )

    first_stage_path: str | None = None
    if args.candidate_source == "run":
        top_ranked, first_stage_path = candidate_source_for_task(
            args=args,
            spec=spec,
            queries=queries,
        )
        corpus = load_rusbeir_corpus(
            spec=spec,
            text_type=args.text_type,
            needed_doc_ids=needed_prefix_doc_ids(top_ranked, rerank_top_k=rerank_top_k),
            max_corpus_docs=args.max_corpus_docs,
        )
    else:
        corpus = load_rusbeir_corpus(
            spec=spec,
            text_type=args.text_type,
            needed_doc_ids=None,
            max_corpus_docs=args.max_corpus_docs,
        )
        top_ranked, first_stage_path = candidate_source_for_task(
            args=args,
            spec=spec,
            queries=queries,
            corpus=corpus,
        )

    first_stage_score_entry = evaluate_results(top_ranked_to_scores(top_ranked), qrels)
    reranked_results, scorer_stats = rerank_with_cross_encoder(
        model=model,
        corpus=corpus,
        queries=queries,
        top_ranked=top_ranked,
        rerank_top_k=rerank_top_k,
        batch_size=args.batch_size,
        show_progress_bar=not args.quiet,
    )
    reranked_score_entry = evaluate_results(reranked_results, qrels)

    run_path = prediction_dir / f"{spec.name}.run"
    json_path = prediction_dir / f"{spec.name}.json"
    write_trec_run(run_path, reranked_results, run_name=sanitize_model_name(args.model_name_or_path))
    write_json_results(json_path, reranked_results)

    task_result = {
        "task_name": spec.name,
        "display_name": spec.display_name,
        "split": split,
        "dataset": {
            "corpus_repo": spec.corpus_repo,
            "qrels_repo": spec.qrels_repo,
            "corpus_documents_loaded": len(corpus),
            "queries_loaded": len(queries),
            "qrels_queries": len(qrels),
            "queries_with_candidates": len(top_ranked),
        },
        "first_stage_path": first_stage_path,
        "scores": {split: [reranked_score_entry]},
        "first_stage_scores": {split: [first_stage_score_entry]},
        "artifacts": {
            "prediction_run": str(run_path.resolve()),
            "prediction_json": str(json_path.resolve()),
        },
    }
    logger.info(
        "Finished rusBEIR cross-encoder task: task=%s, split=%s, ndcg_at_10=%s",
        spec.name,
        split,
        reranked_score_entry.get("ndcg_at_10"),
    )
    return task_result, scorer_stats


def build_summary(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    prediction_dir: Path,
    run_stats_path: Path,
    device: str,
    max_length: int | None,
    rerank_top_k: int,
    max_queries_per_task: int | None,
    task_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "benchmark": RUSBEIR_BENCHMARK_NAME,
        "benchmark_version": RUSBEIR_BENCHMARK_VERSION,
        "model_name_or_path": args.model_name_or_path,
        "revision": args.revision,
        "device": device,
        "batch_size": args.batch_size,
        "max_length": max_length,
        "profile": args.profile,
        "candidate_source": args.candidate_source,
        "first_stage_top_k": args.first_stage_top_k,
        "rerank_top_k": rerank_top_k,
        "max_queries_per_task": max_queries_per_task,
        "max_corpus_docs": args.max_corpus_docs,
        "text_type": args.text_type,
        "tasks": list(task_results),
        "artifacts": {
            "output_dir": str(output_dir.resolve()),
            "prediction_dir": str(prediction_dir.resolve()),
            "summary_path": str((output_dir / "summary.json").resolve()),
            "run_stats_path": str(run_stats_path.resolve()),
        },
        "task_results": task_results,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Model: {summary['model_name_or_path']}")
    print(f"Benchmark: {summary['benchmark']} {summary['benchmark_version']}")
    print(f"Device: {summary['device']}")
    print(
        "Profile: "
        f"{summary['profile']} "
        f"(first_stage_top_k={summary['first_stage_top_k']}, "
        f"rerank_top_k={summary['rerank_top_k']})"
    )
    print(f"Candidate source: {summary['candidate_source']}")
    print(f"Tasks: {', '.join(summary['tasks'])}")
    print(f"Artifacts: {summary['artifacts']['output_dir']}")
    for task_name in summary["tasks"]:
        task_result = summary["task_results"][task_name]
        reranked = task_result["scores"][task_result["split"]][0]
        first_stage = task_result["first_stage_scores"][task_result["split"]][0]
        compact_scores = json.dumps(
            {
                "first_stage_ndcg_at_10": first_stage.get("ndcg_at_10"),
                "ndcg_at_10": reranked.get("ndcg_at_10"),
                "map_at_1000": reranked.get("map_at_1000"),
                "main_score": reranked.get("main_score"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        print(f"{task_name}: {compact_scores}")


def _merge_scorer_stats(stats_by_task: dict[str, dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "queries_processed": 0,
        "queries_reranked": 0,
        "queries_skipped_due_to_missing_docs": 0,
        "documents_scored": 0,
        "candidates_total": 0,
    }
    for stats in stats_by_task.values():
        for key in totals:
            totals[key] += int(stats.get(key, 0))
    totals["tasks"] = stats_by_task
    return totals


def _document_text(document_row: dict[str, Any]) -> str:
    return combine_title_and_text(
        str(document_row.get("title", "") or ""),
        str(document_row.get("text", "") or ""),
    )


def _score_to_float(score: Any) -> float:
    if hasattr(score, "tolist"):
        score = score.tolist()
    if isinstance(score, (list, tuple)):
        if not score:
            return 0.0
        score = score[-1]
    return float(score)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(level=args.log_level, log_file=args.log_file)
    specs = resolve_rusbeir_dataset_specs(args.tasks)
    rerank_top_k, max_queries_per_task = resolve_profile_args(args)
    device = resolve_device(args.device)

    output_dir = resolve_output_dir(args.model_name_or_path, args.output_dir)
    prediction_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting cross-encoder rusBEIR run: model=%s, tasks=%s, candidate_source=%s, "
        "batch_size=%s, device=%s, output_dir=%s",
        args.model_name_or_path,
        [spec.name for spec in specs],
        args.candidate_source,
        args.batch_size,
        device,
        output_dir.resolve(),
    )

    model = CrossEncoder(
        args.model_name_or_path,
        device=device,
        revision=args.revision,
        max_length=args.max_length,
    )
    effective_max_length = model_max_length(model, args.max_length)
    logger.info(
        "CrossEncoder loaded: device=%s, effective_max_length=%s",
        device,
        effective_max_length,
    )

    task_results = {}
    scorer_stats_by_task = {}
    for spec in specs:
        split = split_for_spec(spec, args.split)
        task_result, scorer_stats = run_task(
            args=args,
            spec=spec,
            split=split,
            model=model,
            prediction_dir=prediction_dir,
            rerank_top_k=rerank_top_k,
            max_queries_per_task=max_queries_per_task,
        )
        task_results[spec.name] = task_result
        scorer_stats_by_task[spec.name] = scorer_stats

    run_stats = {
        "model": {
            "model_name_or_path": args.model_name_or_path,
            "revision": args.revision,
            "device": device,
            "batch_size": args.batch_size,
            "max_length": effective_max_length,
        },
        "scorer": _merge_scorer_stats(scorer_stats_by_task),
    }
    run_stats_path = output_dir / "run_stats.json"
    run_stats_path.write_text(
        json.dumps(run_stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = build_summary(
        args=args,
        output_dir=output_dir,
        prediction_dir=prediction_dir,
        run_stats_path=run_stats_path,
        device=device,
        max_length=effective_max_length,
        rerank_top_k=rerank_top_k,
        max_queries_per_task=max_queries_per_task,
        task_results=task_results,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote run stats: %s", run_stats_path.resolve())
    logger.info("Wrote summary: %s", summary_path.resolve())
    print_summary(summary)
    logger.info("Cross-encoder rusBEIR run completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
