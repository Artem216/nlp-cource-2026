from __future__ import annotations

import logging
from typing import Any, Sequence

import requests

from .base import BaseLLMProvider, SchemaUnsupportedError


logger = logging.getLogger(__name__)


class VLLMProvider(BaseLLMProvider):
    provider_name = "vllm"

    def __init__(
        self,
        *,
        model: str,
        request_cache,
        temperature: float,
        timeout: float,
        max_retries: int,
        base_url: str,
        api_key: str | None,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
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
        self.disable_thinking = disable_thinking

    def stats(self) -> dict[str, Any]:
        payload = super().stats()
        payload["disable_thinking"] = self.disable_thinking
        return payload

    def preflight(self) -> None:
        logger.info("vLLM preflight request: base_url=%s", self.base_url)
        response = requests.get(
            f"{self.base_url}/models",
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        logger.info("vLLM preflight succeeded: status_code=%s", response.status_code)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
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
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
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
            "vLLM chat request: model=%s, schema=%s, messages=%s, disable_thinking=%s",
            self.model,
            schema is not None,
            len(messages),
            self.disable_thinking,
        )
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        if schema is not None and response.status_code >= 400 and self._is_schema_error(response):
            logger.warning(
                "vLLM structured output rejected: status_code=%s, body_prefix=%r",
                response.status_code,
                response.text[:300],
            )
            raise SchemaUnsupportedError(response.text)
        response.raise_for_status()
        logger.debug("vLLM chat response: status_code=%s", response.status_code)
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
