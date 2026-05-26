from __future__ import annotations

import logging
from typing import Any, Sequence

import requests

from .base import BaseLLMProvider, SchemaUnsupportedError


logger = logging.getLogger(__name__)


class OpenRouterProvider(BaseLLMProvider):
    provider_name = "openrouter"

    def __init__(
        self,
        *,
        model: str,
        request_cache,
        temperature: float,
        timeout: float,
        max_retries: int,
        base_url: str,
        api_key: str,
        site_url: str | None,
        app_name: str | None,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(
            model=model,
            request_cache=request_cache,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,
        )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.site_url = site_url
        self.app_name = app_name

    def preflight(self) -> None:
        logger.info("OpenRouter preflight request: base_url=%s", self.base_url)
        response = requests.get(
            f"{self.base_url}/models",
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        logger.info("OpenRouter preflight succeeded: status_code=%s", response.status_code)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers

    def _generate_content(
        self,
        *,
        messages: Sequence[dict[str, str]],
        schema: dict[str, Any] | None,
    ) -> tuple[Any, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "reranker_response",
                    "strict": True,
                    "schema": schema,
                },
            }
        logger.debug(
            "OpenRouter chat request: model=%s, schema=%s, messages=%s",
            self.model,
            schema is not None,
            len(messages),
        )
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        if schema is not None and response.status_code >= 400 and self._is_schema_error(response):
            logger.warning(
                "OpenRouter structured output rejected: status_code=%s, body_prefix=%r",
                response.status_code,
                response.text[:300],
            )
            raise SchemaUnsupportedError(response.text)
        response.raise_for_status()
        logger.debug("OpenRouter chat response: status_code=%s", response.status_code)
        data = response.json()
        return self._extract_openai_chat_content(data), data

    @staticmethod
    def _is_schema_error(response: requests.Response) -> bool:
        lowered = response.text.lower()
        hints = (
            "response_format",
            "json_schema",
            "structured output",
            "structured outputs",
            "unsupported",
            "not support",
        )
        return any(hint in lowered for hint in hints)
