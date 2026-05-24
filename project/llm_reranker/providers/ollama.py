from __future__ import annotations

import logging
from typing import Any, Sequence

import requests

from .base import BaseLLMProvider, ProviderError


logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    provider_name = "ollama"

    def __init__(
        self,
        *,
        model: str,
        request_cache,
        temperature: float,
        timeout: float,
        max_retries: int,
        base_url: str,
        keep_alive: str | None,
    ) -> None:
        super().__init__(
            model=model,
            request_cache=request_cache,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive

    def preflight(self) -> None:
        logger.info("Ollama preflight request: base_url=%s", self.base_url)
        response = requests.get(
            f"{self.base_url}/api/tags",
            timeout=self.timeout,
        )
        response.raise_for_status()
        logger.info("Ollama preflight succeeded: status_code=%s", response.status_code)

    def _generate_content(
        self,
        *,
        messages: Sequence[dict[str, str]],
        schema: dict[str, Any] | None,
    ) -> tuple[Any, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if schema is not None:
            payload["format"] = schema
        logger.debug(
            "Ollama chat request: model=%s, schema=%s, messages=%s, keep_alive=%s",
            self.model,
            schema is not None,
            len(messages),
            self.keep_alive,
        )
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        logger.debug("Ollama chat response: status_code=%s", response.status_code)
        data = response.json()
        message = data.get("message") or {}
        content = message.get("content")
        if content is None:
            logger.error("Ollama response is missing message.content")
            raise ProviderError("ollama response does not contain message.content")
        return content, data
