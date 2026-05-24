#!/usr/bin/env python3
"""Run a cross-encoder model on the ruMTEB reranking tasks."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import mteb
import torch
from sentence_transformers import CrossEncoder

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from project.llm_reranker.logging_utils import add_logging_arguments, configure_logging


logger = logging.getLogger(__name__)

BENCHMARK_NAME = "MTEB(rus)"
BENCHMARK_VERSION = "v1.1"
ALLOWED_TASKS = ("MIRACLReranking", "RuBQReranking")
OVERWRITE_STRATEGIES = ("always", "never", "only-missing", "only-cache")
DEVICE_PATTERN = re.compile(r"^(auto|cpu|cuda(?::\d+)?)$")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def device_value(value: str) -> str:
    if not DEVICE_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "device must be one of: auto, cpu, cuda, cuda:N"
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a cross-encoder model on ruMTEB reranking tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_name_or_path", help="Hugging Face model id or local path")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=ALLOWED_TASKS,
        default=list(ALLOWED_TASKS),
        help="Subset of ruMTEB reranking tasks to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cache, predictions, and summary.json. "
            "Defaults to project/results/<sanitized-model-name>"
        ),
    )
    parser.add_argument(
        "--device",
        type=device_value,
        default="auto",
        help="Inference device",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=32,
        help="Batch size passed to CrossEncoder.predict",
    )
    parser.add_argument(
        "--max-length",
        type=positive_int,
        default=None,
        help="Optional max input length passed to CrossEncoder",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision to pass to CrossEncoder",
    )
    parser.add_argument(
        "--overwrite-strategy",
        choices=OVERWRITE_STRATEGIES,
        default="only-missing",
        help="MTEB cache overwrite strategy",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable MTEB and model progress bars",
    )
    add_logging_arguments(parser)
    return parser.parse_args()


def sanitize_model_name(model_name_or_path: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name_or_path).strip("-._")
    return sanitized or "model"


def resolve_output_dir(model_name_or_path: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        logger.debug("Using explicit output directory: %s", output_dir)
        return output_dir
    script_dir = Path(__file__).resolve().parent
    resolved = script_dir / "results" / sanitize_model_name(model_name_or_path)
    logger.debug("Resolved default output directory: %s", resolved)
    return resolved


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        logger.debug("Using explicitly requested device: %s", requested_device)
        return requested_device
    resolved = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Auto-selected inference device: %s", resolved)
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

    # Guard against future MTEB changes that could widen the selected language subsets.
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

    logger.info("Loaded tasks: %s", [task.metadata.name for task in tasks])
    return tasks


def model_max_length(model: CrossEncoder, requested_max_length: int | None) -> int | None:
    if requested_max_length is not None:
        return requested_max_length
    getter = getattr(model, "get_max_seq_length", None)
    if callable(getter):
        value = getter()
        if isinstance(value, int):
            return value
    value = getattr(model, "max_seq_length", None)
    if isinstance(value, int):
        return value
    return None


def serialize_task_result(task_result: Any) -> dict[str, Any]:
    if hasattr(task_result, "model_dump"):
        return task_result.model_dump(mode="json")
    if hasattr(task_result, "to_dict"):
        return task_result.to_dict()
    raise TypeError(f"Unsupported task result type: {type(task_result)!r}")


def build_summary(
    *,
    model_name_or_path: str,
    revision: str | None,
    device: str,
    batch_size: int,
    max_length: int | None,
    tasks: Sequence[Any],
    output_dir: Path,
    results: Any,
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
        "model_name_or_path": model_name_or_path,
        "revision": revision,
        "device": device,
        "batch_size": batch_size,
        "max_length": max_length,
        "tasks": [task.metadata.name for task in tasks],
        "artifacts": {
            "output_dir": str(output_dir.resolve()),
            "cache_dir": str((output_dir / "cache").resolve()),
            "prediction_dir": str((output_dir / "predictions").resolve()),
            "summary_path": str(summary_path.resolve()),
        },
        "task_results": task_results,
    }


def print_summary(summary: dict[str, Any]) -> None:
    logger.info("Printing compact run summary")
    print(f"Model: {summary['model_name_or_path']}")
    print(f"Benchmark: {summary['benchmark']} {summary['benchmark_version']}")
    print(f"Device: {summary['device']}")
    print(f"Tasks: {', '.join(summary['tasks'])}")
    print(f"Artifacts: {summary['artifacts']['output_dir']}")
    for task_name in summary["tasks"]:
        task_result = summary["task_results"][task_name]
        compact_scores = json.dumps(
            task_result["scores"],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        print(f"{task_name}: {compact_scores}")


def main() -> int:
    args = parse_args()
    configure_logging(level=args.log_level, log_file=args.log_file)
    logger.info(
        "Starting cross-encoder ruMTEB run: model=%s, tasks=%s, batch_size=%s, "
        "max_length=%s, overwrite_strategy=%s",
        args.model_name_or_path,
        args.tasks,
        args.batch_size,
        args.max_length,
        args.overwrite_strategy,
    )

    tasks = load_tasks(args.tasks)
    device = resolve_device(args.device)
    output_dir = resolve_output_dir(args.model_name_or_path, args.output_dir)
    cache_dir = output_dir / "cache"
    prediction_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Prepared artifact directories: output=%s, cache=%s, predictions=%s",
        output_dir.resolve(),
        cache_dir.resolve(),
        prediction_dir.resolve(),
    )

    logger.info("Loading CrossEncoder model: %s", args.model_name_or_path)
    model = CrossEncoder(
        args.model_name_or_path,
        device=device,
        revision=args.revision,
        max_length=args.max_length,
    )
    logger.info(
        "CrossEncoder loaded: device=%s, effective_max_length=%s",
        device,
        model_max_length(model, args.max_length),
    )

    logger.info("Starting MTEB evaluation")
    results = mteb.evaluate(
        model,
        tasks=tasks,
        raise_error=True,
        encode_kwargs={
            "batch_size": args.batch_size,
            "show_progress_bar": not args.quiet,
        },
        cache=mteb.ResultCache(cache_path=cache_dir),
        overwrite_strategy=args.overwrite_strategy,
        prediction_folder=prediction_dir,
        show_progress_bar=not args.quiet,
    )
    logger.info("MTEB evaluation finished")

    summary = build_summary(
        model_name_or_path=args.model_name_or_path,
        revision=args.revision,
        device=device,
        batch_size=args.batch_size,
        max_length=model_max_length(model, args.max_length),
        tasks=tasks,
        output_dir=output_dir,
        results=results,
    )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote summary: %s", summary_path.resolve())
    print_summary(summary)
    logger.info("Cross-encoder ruMTEB run completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
