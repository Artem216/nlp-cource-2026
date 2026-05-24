#!/usr/bin/env python3
"""Compare ruMTEB reranking run summaries."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from project.llm_reranker.logging_utils import add_logging_arguments, configure_logging
from project.llm_reranker.utils import first_score_entry


logger = logging.getLogger(__name__)

TASK_COLUMNS = (
    ("MIRACLReranking", "ndcg_at_10"),
    ("MIRACLReranking", "map_at_1000"),
    ("RuBQReranking", "ndcg_at_10"),
    ("RuBQReranking", "map_at_1000"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a markdown or CSV comparison for ruMTEB reranking summaries."
    )
    parser.add_argument("paths", nargs="+", help="summary.json paths or directories containing summary.json")
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    add_logging_arguments(parser)
    return parser.parse_args(argv)


def resolve_summary_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_dir():
        path = path / "summary.json"
    if not path.exists():
        logger.error("Summary path does not exist: %s", path)
        raise FileNotFoundError(path)
    logger.debug("Resolved summary path: %s -> %s", raw_path, path)
    return path


def load_summary(path: Path) -> dict[str, Any]:
    logger.info("Loading summary: %s", path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    logger.debug(
        "Loaded summary: label=%s, tasks=%s",
        run_label(summary),
        summary.get("tasks"),
    )
    return summary


def run_label(summary: dict[str, Any]) -> str:
    provider = summary.get("provider")
    strategy = summary.get("strategy")
    model = summary.get("model_name_or_path", "unknown-model")
    if provider and strategy:
        return f"{provider}/{model} [{strategy}]"
    return model


def metric_value(summary: dict[str, Any], task_name: str, metric: str) -> str:
    task_result = summary.get("task_results", {}).get(task_name)
    if not task_result:
        logger.debug("Missing task result: run=%s, task=%s", run_label(summary), task_name)
        return ""
    try:
        score_entry = first_score_entry(task_result["scores"])
    except Exception:  # noqa: BLE001
        logger.exception(
            "Could not read score entry: run=%s, task=%s",
            run_label(summary),
            task_name,
        )
        return ""
    value = score_entry.get(metric)
    if value is None:
        logger.debug(
            "Missing metric: run=%s, task=%s, metric=%s",
            run_label(summary),
            task_name,
            metric,
        )
        return ""
    return f"{value:.5f}"


def build_rows(summaries: Sequence[dict[str, Any]]) -> list[list[str]]:
    logger.info("Building comparison rows for %d summaries", len(summaries))
    header = ["run"] + [f"{task}:{metric}" for task, metric in TASK_COLUMNS]
    rows = [header]
    for summary in summaries:
        row = [run_label(summary)]
        for task_name, metric in TASK_COLUMNS:
            row.append(metric_value(summary, task_name, metric))
        rows.append(row)
    return rows


def print_markdown(rows: Sequence[Sequence[str]]) -> None:
    logger.debug("Printing markdown comparison with %d data rows", max(len(rows) - 1, 0))
    header = rows[0]
    separator = ["---"] * len(header)
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(separator) + " |")
    for row in rows[1:]:
        print("| " + " | ".join(row) + " |")


def print_csv(rows: Sequence[Sequence[str]]) -> None:
    logger.debug("Printing CSV comparison with %d data rows", max(len(rows) - 1, 0))
    writer = csv.writer(sys.stdout)
    writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(level=args.log_level, log_file=args.log_file)
    logger.info("Starting ruMTEB comparison: format=%s, paths=%s", args.format, args.paths)
    summaries = [load_summary(resolve_summary_path(raw_path)) for raw_path in args.paths]
    rows = build_rows(summaries)
    if args.format == "csv":
        print_csv(rows)
    else:
        print_markdown(rows)
    logger.info("Comparison completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
