from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import mteb

from .utils import first_score_entry, sanitize_model_name


logger = logging.getLogger(__name__)

BENCHMARK_NAME = "MTEB(rus)"
BENCHMARK_VERSION = "v1.1"
ALLOWED_TASKS = ("MIRACLReranking", "RuBQReranking")
OVERWRITE_STRATEGIES = ("always", "never", "only-missing", "only-cache")
PROFILE_DEFAULTS = {
    "full": {"rerank_top_k": 20, "max_queries_per_task": None},
    "quick": {"rerank_top_k": 10, "max_queries_per_task": 25},
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive number")
    return parsed


def sanitize_run_name(provider: str, model: str, strategy: str) -> str:
    return sanitize_model_name(f"{provider}_{model}_{strategy}")


def resolve_output_dir(
    *,
    provider: str,
    model_name_or_path: str,
    strategy: str,
    output_dir: Path | None,
) -> Path:
    if output_dir is not None:
        logger.debug("Using explicit output directory: %s", output_dir)
        return output_dir
    script_dir = Path(__file__).resolve().parent.parent
    date_part = datetime.now().date().isoformat()
    resolved = script_dir / "results" / f"runs_{date_part}" / sanitize_run_name(
        provider,
        model_name_or_path,
        strategy,
    )
    logger.debug("Resolved default output directory: %s", resolved)
    return resolved


def load_tasks(selected_task_names: Sequence[str]) -> list[Any]:
    logger.info("Loading ruMTEB reranking tasks: requested=%s", list(selected_task_names))
    available_tasks = mteb.get_tasks(
        languages=["rus"],
        task_types=["Reranking"],
        exclude_superseded=True,
        exclude_private=True,
    )
    task_lookup = {
        task.metadata.name: task
        for task in available_tasks
        if task.metadata.name in ALLOWED_TASKS
    }
    missing = [task_name for task_name in ALLOWED_TASKS if task_name not in task_lookup]
    if missing:
        missing_text = ", ".join(missing)
        logger.error("Installed MTEB does not expose expected tasks: %s", missing_text)
        raise RuntimeError(
            "The installed mteb version does not expose the expected ruMTEB reranking "
            f"tasks: {missing_text}"
        )

    requested = list(dict.fromkeys(selected_task_names))
    tasks = [task_lookup[task_name] for task_name in requested]
    for task in tasks:
        subsets = getattr(task, "hf_subsets", None)
        if task.metadata.name == "MIRACLReranking" and subsets != ["ru"]:
            raise RuntimeError(
                "Expected MIRACLReranking to be restricted to the Russian subset, "
                f"got {subsets!r}"
            )
        if task.metadata.name == "RuBQReranking" and subsets != ["default"]:
            raise RuntimeError(
                "Expected RuBQReranking to use the default subset, "
                f"got {subsets!r}"
            )
        logger.debug(
            "Task subset verified: task=%s, hf_subsets=%s",
            task.metadata.name,
            subsets,
        )
    logger.info("Loaded tasks: %s", [task.metadata.name for task in tasks])
    return tasks


def subset_task_queries(tasks: Sequence[Any], max_queries_per_task: int | None) -> None:
    if max_queries_per_task is None:
        logger.info("Query subsetting disabled; using full task datasets")
        return
    logger.info("Limiting task queries: max_queries_per_task=%s", max_queries_per_task)
    for task in tasks:
        logger.info("Loading task data for subsetting: task=%s", task.metadata.name)
        task.load_data()
        for subset_name, split_map in task.dataset.items():
            for split_name, split_data in split_map.items():
                queries = split_data["queries"]
                if len(queries) <= max_queries_per_task:
                    logger.debug(
                        "No query limit needed: task=%s, subset=%s, split=%s, queries=%s",
                        task.metadata.name,
                        subset_name,
                        split_name,
                        len(queries),
                    )
                    continue
                limited_queries = queries.select(range(max_queries_per_task))
                allowed_query_ids = set(limited_queries["id"])
                relevant_docs = {
                    query_id: docs
                    for query_id, docs in split_data["relevant_docs"].items()
                    if query_id in allowed_query_ids
                }
                top_ranked = split_data.get("top_ranked")
                limited_top_ranked = None
                if top_ranked is not None:
                    limited_top_ranked = {
                        query_id: top_ranked[query_id]
                        for query_id in limited_queries["id"]
                        if query_id in top_ranked
                    }

                needed_doc_ids: set[str] = set()
                source = limited_top_ranked if limited_top_ranked is not None else relevant_docs
                for values in source.values():
                    if isinstance(values, dict):
                        needed_doc_ids.update(values.keys())
                    else:
                        needed_doc_ids.update(values)

                corpus = split_data["corpus"]
                keep_indices = [
                    index
                    for index, doc_id in enumerate(corpus["id"])
                    if doc_id in needed_doc_ids
                ]
                limited_corpus = corpus.select(keep_indices)
                task.dataset[subset_name][split_name] = {
                    "corpus": limited_corpus,
                    "queries": limited_queries,
                    "relevant_docs": relevant_docs,
                    "top_ranked": limited_top_ranked,
                }
                logger.info(
                    "Subsetted task split: task=%s, subset=%s, split=%s, "
                    "queries=%s->%s, corpus=%s->%s",
                    task.metadata.name,
                    subset_name,
                    split_name,
                    len(queries),
                    len(limited_queries),
                    len(corpus),
                    len(limited_corpus),
                )
        task.data_loaded = True
        logger.debug("Marked task as loaded after subsetting: task=%s", task.metadata.name)


def serialize_task_result(task_result: Any) -> dict[str, Any]:
    if hasattr(task_result, "model_dump"):
        return task_result.model_dump(mode="json")
    if hasattr(task_result, "to_dict"):
        return task_result.to_dict()
    raise TypeError(f"Unsupported task result type: {type(task_result)!r}")


def build_summary(
    *,
    provider: str,
    model_name_or_path: str,
    strategy: str,
    profile: str,
    rerank_top_k: int,
    max_queries_per_task: int | None,
    temperature: float,
    timeout: float,
    concurrency: int,
    prompt_language: str,
    tasks: Sequence[Any],
    output_dir: Path,
    results: Any,
    run_stats_path: Path,
) -> dict[str, Any]:
    logger.debug("Serializing %d task results", len(results.task_results))
    task_results = {}
    for task_result in results.task_results:
        serialized = serialize_task_result(task_result)
        task_results[serialized["task_name"]] = serialized

    summary_path = output_dir / "summary.json"
    return {
        "benchmark": BENCHMARK_NAME,
        "benchmark_version": BENCHMARK_VERSION,
        "provider": provider,
        "model_name_or_path": model_name_or_path,
        "strategy": strategy,
        "profile": profile,
        "rerank_top_k": rerank_top_k,
        "max_queries_per_task": max_queries_per_task,
        "temperature": temperature,
        "timeout": timeout,
        "concurrency": concurrency,
        "prompt_language": prompt_language,
        "tasks": [task.metadata.name for task in tasks],
        "artifacts": {
            "output_dir": str(output_dir.resolve()),
            "cache_dir": str((output_dir / "cache").resolve()),
            "prediction_dir": str((output_dir / "predictions").resolve()),
            "request_cache_dir": str((output_dir / "request_cache").resolve()),
            "summary_path": str(summary_path.resolve()),
            "run_stats_path": str(run_stats_path.resolve()),
        },
        "task_results": task_results,
    }


def print_summary(summary: dict[str, Any]) -> None:
    logger.info("Printing compact run summary")
    print(f"Model: {summary['provider']}::{summary['model_name_or_path']}")
    print(f"Strategy: {summary['strategy']}")
    print(f"Benchmark: {summary['benchmark']} {summary['benchmark_version']}")
    print(f"Profile: {summary['profile']} (rerank_top_k={summary['rerank_top_k']})")
    print(f"Tasks: {', '.join(summary['tasks'])}")
    print(f"Artifacts: {summary['artifacts']['output_dir']}")
    for task_name in summary["tasks"]:
        score_entry = first_score_entry(summary["task_results"][task_name]["scores"])
        compact_scores = json.dumps(
            {
                "ndcg_at_10": score_entry.get("ndcg_at_10"),
                "map_at_1000": score_entry.get("map_at_1000"),
                "main_score": score_entry.get("main_score"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        print(f"{task_name}: {compact_scores}")
