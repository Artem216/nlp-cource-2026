from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"


def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default=os.environ.get("RUMTEB_LOG_LEVEL", "INFO").upper(),
        help="Logging verbosity. Can also be set with RUMTEB_LOG_LEVEL.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional file path that receives the same logs as stderr.",
    )


def configure_logging(*, level: str, log_file: Path | None = None) -> None:
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unsupported log level: {level}")

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format=DEFAULT_LOG_FORMAT,
        handlers=handlers,
        force=True,
    )
    logging.captureWarnings(True)

    # Keep noisy HTTP internals quiet unless the whole run is explicitly DEBUG.
    if numeric_level > logging.DEBUG:
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.getLogger(__name__).debug(
        "Logging configured: level=%s, log_file=%s",
        level.upper(),
        str(log_file) if log_file else None,
    )
