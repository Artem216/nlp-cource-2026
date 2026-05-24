from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=None)
def load_prompt_template(strategy_name: str, prompt_language: str, role: str) -> str:
    path = PROMPT_DIR / strategy_name / f"{prompt_language}_{role}.txt"
    logger.debug("Loading prompt template: %s", path)
    return path.read_text(encoding="utf-8").strip()


def render_prompt_template(
    strategy_name: str,
    prompt_language: str,
    role: str,
    **values: Any,
) -> str:
    template = load_prompt_template(strategy_name, prompt_language, role)
    if not values:
        return template
    return template.format(**values)
