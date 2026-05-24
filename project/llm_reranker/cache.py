from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class RequestCache:
    """Small thread-safe JSON cache for expensive provider calls."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._writes = 0
        logger.info("Request cache ready: dir=%s", self.root_dir.resolve())

    def _path_for_key(self, key: str) -> Path:
        return self.root_dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path_for_key(key)
        if not path.exists():
            with self._lock:
                self._misses += 1
                misses = self._misses
            logger.debug(
                "Request cache miss: key=%s, path=%s, misses=%s",
                key[:12],
                path,
                misses,
            )
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        with self._lock:
            self._hits += 1
            hits = self._hits
        logger.debug(
            "Request cache hit: key=%s, path=%s, hits=%s",
            key[:12],
            path,
            hits,
        )
        return payload

    def set(self, key: str, payload: dict[str, Any]) -> None:
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")
        serializable = dict(payload)
        serializable.setdefault(
            "cached_at",
            datetime.now(tz=timezone.utc).isoformat(),
        )
        temp_path.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
        with self._lock:
            self._writes += 1
            writes = self._writes
        logger.debug(
            "Request cache write: key=%s, path=%s, writes=%s",
            key[:12],
            path,
            writes,
        )

    def stats(self) -> dict[str, int | str]:
        with self._lock:
            hits = self._hits
            misses = self._misses
            writes = self._writes
        return {
            "cache_dir": str(self.root_dir.resolve()),
            "hits": hits,
            "misses": misses,
            "writes": writes,
        }
