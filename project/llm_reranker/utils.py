from __future__ import annotations

import json
import logging
import re
from hashlib import sha256
from typing import Any, Iterable


logger = logging.getLogger(__name__)

WHITESPACE_RE = re.compile(r"\s+")


def sanitize_model_name(model_name_or_path: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name_or_path).strip("-._")
    return sanitized or "model"


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def truncate_text(text: str, max_chars: int | None) -> str:
    normalized = normalize_whitespace(text)
    if max_chars is None or max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    if max_chars <= 1:
        logger.debug("Truncated text to %s char from %s chars", max_chars, len(normalized))
        return normalized[:max_chars]
    logger.debug("Truncated text to %s chars from %s chars", max_chars, len(normalized))
    return normalized[: max_chars - 1].rstrip() + "…"


def combine_title_and_text(title: str | None, text: str | None) -> str:
    title = normalize_whitespace(title or "")
    text = normalize_whitespace(text or "")
    if title and text:
        return f"{title}\n\n{text}"
    return title or text


def json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def extract_json_from_text(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        logger.error("Provider response was empty while JSON was expected")
        raise ValueError("expected non-empty JSON payload")

    decoder = json.JSONDecoder()
    try:
        parsed, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        parsed = None
    else:
        if stripped[end:].strip():
            parsed = None
        else:
            logger.debug("Parsed provider response as direct JSON")
            return parsed

    candidate_positions = [idx for idx, ch in enumerate(stripped) if ch in "[{"]
    for start in candidate_positions:
        try:
            parsed, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        logger.debug("Parsed provider response JSON from offset %s", start)
        return parsed

    logger.error("Failed to parse JSON from provider response: prefix=%r", stripped[:300])
    raise ValueError("failed to parse JSON from provider response")


def make_descending_scores(doc_ids: Iterable[str]) -> dict[str, float]:
    ordered = list(doc_ids)
    total = len(ordered)
    return {doc_id: float(total - index) for index, doc_id in enumerate(ordered)}


def first_score_entry(scores: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    for split_entries in scores.values():
        if split_entries:
            logger.debug("Selected first score entry from split with %d entries", len(split_entries))
            return split_entries[0]
    logger.error("Task result does not contain score entries")
    raise ValueError("task result does not contain score entries")
