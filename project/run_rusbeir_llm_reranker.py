#!/usr/bin/env python3
"""Run LLM-as-reranker strategies on rusBEIR datasets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from project.llm_reranker.benchmark import (  # noqa: E402
    OVERWRITE_STRATEGIES,
    PROFILE_DEFAULTS,
    positive_float,
    positive_int,
    sanitize_run_name,
)
from project.llm_reranker.cache import RequestCache  # noqa: E402
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
from project.llm_reranker.search_adapter import LLMRerankerSearchModel  # noqa: E402
from project.run_rumteb_llm_reranker import (  # noqa: E402
    PROMPT_LANGUAGES,
    PROVIDERS,
    STRATEGIES,
    create_provider,
    create_strategy,
    resolve_profile_args,
)


logger = logging.getLogger(__name__)

CANDIDATE_SOURCES = ("auto", "run", "tfidf")
SPLITS = ("auto", "train", "dev", "test")
TEXT_TYPES = ("processed_text", "text")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-reranker strategies on rusBEIR datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--provider", required=True, choices=PROVIDERS)
    parser.add_argument("--model", required=True, help="Provider model identifier")
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
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
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for cache, predictions, and summaries.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_DEFAULTS),
        default="quick",
        help="Benchmark profile controlling default query and rerank limits.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=positive_int,
        default=None,
        help="Number of top-ranked candidates to rerank with the LLM.",
    )
    parser.add_argument(
        "--max-queries-per-task",
        type=positive_int,
        default=None,
        help="Optional query cap per dataset.",
    )
    parser.add_argument(
        "--overwrite-strategy",
        choices=OVERWRITE_STRATEGIES,
        default="only-missing",
        help="Recorded in summary for parity with ruMTEB runs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature passed to the provider.",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=210.0,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--concurrency",
        type=positive_int,
        default=2,
        help="Number of query-level reranking workers.",
    )
    parser.add_argument(
        "--max-retries",
        type=positive_int,
        default=3,
        help="Number of provider retries for malformed/transient responses.",
    )
    parser.add_argument(
        "--max-tokens",
        type=positive_int,
        default=512,
        help="Maximum completion tokens for each reranker JSON response.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help=(
            "Ask vLLM chat templates that support it to disable thinking/reasoning "
            "tokens. Useful for Qwen thinking models that return null content."
        ),
    )
    parser.add_argument(
        "--prompt-language",
        choices=PROMPT_LANGUAGES,
        default="ru",
        help="Language used in reranking prompts.",
    )
    parser.add_argument(
        "--query-max-chars",
        type=positive_int,
        default=500,
        help="Maximum query length included in prompts.",
    )
    parser.add_argument(
        "--doc-max-chars",
        type=positive_int,
        default=None,
        help="Override strategy-specific document truncation limit.",
    )
    parser.add_argument(
        "--window-size",
        type=positive_int,
        default=10,
        help="Sliding window size for listwise reranking.",
    )
    parser.add_argument(
        "--stride",
        type=positive_int,
        default=5,
        help="Sliding window stride for listwise reranking.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Provider base URL. Defaults to OLLAMA_HOST, OpenRouter API, or local vLLM API.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter/vLLM API key. Defaults to OPENROUTER_API_KEY or VLLM_API_KEY.",
    )
    parser.add_argument("--keep-alive", default=None, help="Optional Ollama keep_alive value.")
    parser.add_argument(
        "--site-url",
        default=None,
        help="Optional HTTP-Referer header for OpenRouter requests.",
    )
    parser.add_argument(
        "--app-name",
        default="rusBEIR LLM Reranker",
        help="Optional X-Title header for OpenRouter requests.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip provider connectivity validation before evaluation.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce progress output.")
    add_logging_arguments(parser)
    args = parser.parse_args(argv)
    if args.candidate_source == "auto":
        args.candidate_source = "run" if args.first_stage_run is not None else "tfidf"
    return args


def resolve_output_dir(
    *,
    provider: str,
    model_name_or_path: str,
    strategy: str,
    output_dir: Path | None,
) -> Path:
    if output_dir is not None:
        return output_dir
    script_dir = Path(__file__).resolve().parent
    date_part = datetime.now().date().isoformat()
    return script_dir / "results" / f"rusbeir_runs_{date_part}" / sanitize_run_name(
        provider,
        model_name_or_path,
        strategy,
    )


def candidate_source_for_task(
    *,
    args: argparse.Namespace,
    spec: RusBeirDatasetSpec,
    split: str,
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
    del split
    top_ranked = build_tfidf_first_stage(
        corpus=corpus,
        queries=queries,
        top_k=args.first_stage_top_k,
        max_features=args.tfidf_max_features,
    )
    return top_ranked, None


def evaluate_first_stage(top_ranked: dict[str, list[str]], qrels: dict[str, dict[str, int]]):
    return score_entry_from_metrics(
        evaluate_retrieval(qrels, top_ranked_to_scores(top_ranked), k_values=RUSBEIR_K_VALUES)
    )


def evaluate_reranked(results: dict[str, dict[str, float]], qrels: dict[str, dict[str, int]]):
    return score_entry_from_metrics(evaluate_retrieval(qrels, results, k_values=RUSBEIR_K_VALUES))


def build_summary(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    request_cache_dir: Path,
    prediction_dir: Path,
    run_stats_path: Path,
    rerank_top_k: int,
    max_queries_per_task: int | None,
    task_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "benchmark": RUSBEIR_BENCHMARK_NAME,
        "benchmark_version": RUSBEIR_BENCHMARK_VERSION,
        "provider": args.provider,
        "model_name_or_path": args.model,
        "strategy": args.strategy,
        "profile": args.profile,
        "candidate_source": args.candidate_source,
        "first_stage_top_k": args.first_stage_top_k,
        "rerank_top_k": rerank_top_k,
        "max_queries_per_task": max_queries_per_task,
        "max_corpus_docs": args.max_corpus_docs,
        "text_type": args.text_type,
        "temperature": args.temperature,
        "timeout": args.timeout,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "disable_thinking": args.disable_thinking,
        "prompt_language": args.prompt_language,
        "overwrite_strategy": args.overwrite_strategy,
        "tasks": list(task_results),
        "artifacts": {
            "output_dir": str(output_dir.resolve()),
            "prediction_dir": str(prediction_dir.resolve()),
            "request_cache_dir": str(request_cache_dir.resolve()),
            "summary_path": str((output_dir / "summary.json").resolve()),
            "run_stats_path": str(run_stats_path.resolve()),
        },
        "task_results": task_results,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Model: {summary['provider']}::{summary['model_name_or_path']}")
    print(f"Strategy: {summary['strategy']}")
    print(f"Benchmark: {summary['benchmark']} {summary['benchmark_version']}")
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


def run_task(
    *,
    args: argparse.Namespace,
    spec: RusBeirDatasetSpec,
    split: str,
    model: LLMRerankerSearchModel,
    prediction_dir: Path,
    rerank_top_k: int,
    max_queries_per_task: int | None,
) -> dict[str, Any]:
    logger.info("Starting rusBEIR task: task=%s, split=%s", spec.name, split)
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
            split=split,
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
            split=split,
            queries=queries,
            corpus=corpus,
        )

    first_stage_score_entry = evaluate_first_stage(top_ranked, qrels)
    task_metadata = SimpleNamespace(name=spec.display_name)
    model.index(
        corpus,
        task_metadata=task_metadata,
        hf_split=split,
        hf_subset=spec.name,
        encode_kwargs={},
        num_proc=None,
    )
    reranked_results = model.search(
        queries,
        task_metadata=task_metadata,
        hf_split=split,
        hf_subset=spec.name,
        top_k=args.first_stage_top_k,
        encode_kwargs={},
        top_ranked=top_ranked,
        num_proc=None,
    )
    reranked_score_entry = evaluate_reranked(reranked_results, qrels)

    run_path = prediction_dir / f"{spec.name}.run"
    json_path = prediction_dir / f"{spec.name}.json"
    write_trec_run(run_path, reranked_results, run_name=sanitize_run_name(args.provider, args.model, args.strategy))
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
        "Finished rusBEIR task: task=%s, split=%s, ndcg_at_10=%s",
        spec.name,
        split,
        reranked_score_entry.get("ndcg_at_10"),
    )
    return task_result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(level=args.log_level, log_file=args.log_file)
    specs = resolve_rusbeir_dataset_specs(args.tasks)
    rerank_top_k, max_queries_per_task = resolve_profile_args(args)

    output_dir = resolve_output_dir(
        provider=args.provider,
        model_name_or_path=args.model,
        strategy=args.strategy,
        output_dir=args.output_dir,
    )
    prediction_dir = output_dir / "predictions"
    request_cache_dir = output_dir / "request_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    request_cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting LLM rusBEIR run: provider=%s, model=%s, strategy=%s, tasks=%s, "
        "candidate_source=%s, output_dir=%s",
        args.provider,
        args.model,
        args.strategy,
        [spec.name for spec in specs],
        args.candidate_source,
        output_dir.resolve(),
    )

    request_cache = RequestCache(request_cache_dir)
    provider = create_provider(args, request_cache)
    if not args.skip_preflight:
        provider.preflight()
    else:
        logger.info("Provider preflight skipped")
    strategy = create_strategy(args, provider)
    model = LLMRerankerSearchModel(
        provider_name=args.provider,
        provider_model_name=args.model,
        strategy=strategy,
        rerank_top_k=rerank_top_k,
        concurrency=args.concurrency,
    )

    task_results = {}
    for spec in specs:
        split = split_for_spec(spec, args.split)
        task_results[spec.name] = run_task(
            args=args,
            spec=spec,
            split=split,
            model=model,
            prediction_dir=prediction_dir,
            rerank_top_k=rerank_top_k,
            max_queries_per_task=max_queries_per_task,
        )

    run_stats = {
        "provider": provider.stats(),
        "strategy": strategy.stats(),
        "search_model": model.stats(),
        "request_cache": request_cache.stats(),
    }
    run_stats_path = output_dir / "run_stats.json"
    run_stats_path.write_text(
        json.dumps(run_stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = build_summary(
        args=args,
        output_dir=output_dir,
        request_cache_dir=request_cache_dir,
        prediction_dir=prediction_dir,
        run_stats_path=run_stats_path,
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
    logger.info("LLM rusBEIR run completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
