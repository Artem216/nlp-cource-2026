from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from ..cache import RequestCache
from ..utils import extract_json_from_text, json_hash


logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Raised when a provider request cannot be completed."""


class SchemaUnsupportedError(ProviderError):
    """Raised when the provider/model rejects structured outputs."""


class BaseLLMProvider(ABC):
    provider_name = "provider"

    def __init__(
        self,
        *,
        model: str,
        request_cache: RequestCache | None,
        temperature: float,
        timeout: float,
        max_retries: int,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.request_cache = request_cache
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self._lock = threading.Lock()
        self._requests = 0
        self._schema_fallbacks = 0
        self._retries = 0
        logger.info(
            "Provider initialized: provider=%s, model=%s, temperature=%s, "
            "timeout=%s, max_retries=%s, max_tokens=%s, cache_enabled=%s",
            self.provider_name,
            self.model,
            self.temperature,
            self.timeout,
            self.max_retries,
            self.max_tokens,
            self.request_cache is not None,
        )

    def preflight(self) -> None:
        """Validate provider connectivity."""

    def stats(self) -> dict[str, Any]:
        with self._lock:
            requests_count = self._requests
            schema_fallbacks = self._schema_fallbacks
            retries = self._retries
        return {
            "provider": self.provider_name,
            "model": self.model,
            "requests": requests_count,
            "schema_fallbacks": schema_fallbacks,
            "retries": retries,
            "max_tokens": self.max_tokens,
        }

    def generate_json(
        self,
        *,
        cache_namespace: str,
        messages: Sequence[dict[str, str]],
        schema: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        metadata = metadata or {}
        request_payload = {
            "provider": self.provider_name,
            "model": self.model,
            "temperature": self.temperature,
            "namespace": cache_namespace,
            "schema": schema,
            "messages": list(messages),
            "metadata": metadata,
        }
        cache_key = json_hash(request_payload)
        prompt_chars, prompt_bytes, role_chars = self._prompt_length_stats(messages)
        logger.debug(
            "Provider JSON request prepared: provider=%s, model=%s, namespace=%s, "
            "schema=%s, metadata=%s, prompt_chars=%s, prompt_bytes=%s, "
            "role_chars=%s, cache_key=%s",
            self.provider_name,
            self.model,
            cache_namespace,
            schema is not None,
            metadata,
            prompt_chars,
            prompt_bytes,
            role_chars,
            cache_key[:12],
        )
        if self.request_cache is not None:
            cached = self.request_cache.get(cache_key)
            if cached is not None:
                logger.debug(
                    "Provider JSON request served from cache: namespace=%s, cache_key=%s",
                    cache_namespace,
                    cache_key[:12],
                )
                return cached["parsed"]

        effective_schema = schema
        effective_messages = list(messages)
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            started_at = time.perf_counter()
            try:
                prompt_chars, prompt_bytes, role_chars = self._prompt_length_stats(
                    effective_messages,
                )
                logger.info(
                    "Provider JSON request prompt length: provider=%s, model=%s, "
                    "namespace=%s, attempt=%s/%s, schema=%s, messages=%s, "
                    "prompt_chars=%s, prompt_bytes=%s, role_chars=%s",
                    self.provider_name,
                    self.model,
                    cache_namespace,
                    attempt,
                    self.max_retries,
                    effective_schema is not None,
                    len(effective_messages),
                    prompt_chars,
                    prompt_bytes,
                    role_chars,
                )
                logger.debug(
                    "Provider JSON request attempt: provider=%s, namespace=%s, "
                    "attempt=%s/%s, schema=%s",
                    self.provider_name,
                    cache_namespace,
                    attempt,
                    self.max_retries,
                    effective_schema is not None,
                )
                content, response_payload = self._generate_content(
                    messages=effective_messages,
                    schema=effective_schema,
                )
                parsed = self._coerce_json_payload(content)
                if self.request_cache is not None:
                    self.request_cache.set(
                        cache_key,
                        {
                            "provider": self.provider_name,
                            "model": self.model,
                            "namespace": cache_namespace,
                            "messages": effective_messages,
                            "schema": effective_schema,
                            "metadata": metadata,
                            "parsed": parsed,
                            "raw_content": content,
                            "response_payload": response_payload,
                        },
                    )
                with self._lock:
                    self._requests += 1
                    requests_count = self._requests
                elapsed = time.perf_counter() - started_at
                logger.debug(
                    "Provider JSON request completed: provider=%s, namespace=%s, "
                    "attempt=%s, elapsed=%.2fs, total_requests=%s",
                    self.provider_name,
                    cache_namespace,
                    attempt,
                    elapsed,
                    requests_count,
                )
                return parsed
            except SchemaUnsupportedError as exc:
                last_error = exc
                if effective_schema is None:
                    break
                effective_schema = None
                effective_messages = self._force_json_only_messages(messages)
                with self._lock:
                    self._schema_fallbacks += 1
                    schema_fallbacks = self._schema_fallbacks
                logger.warning(
                    "Provider rejected structured output; falling back to JSON-only prompt: "
                    "provider=%s, namespace=%s, schema_fallbacks=%s, error=%s",
                    self.provider_name,
                    cache_namespace,
                    schema_fallbacks,
                    exc,
                )
                continue
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.max_retries:
                    break
                with self._lock:
                    self._retries += 1
                    retries = self._retries
                sleep_seconds = min(2 ** (attempt - 1), 8)
                logger.warning(
                    "Provider JSON request failed; retrying: provider=%s, "
                    "namespace=%s, attempt=%s/%s, sleep=%ss, retries=%s, error=%s",
                    self.provider_name,
                    cache_namespace,
                    attempt,
                    self.max_retries,
                    sleep_seconds,
                    retries,
                    exc,
                )
                time.sleep(sleep_seconds)

        logger.error(
            "Provider JSON request failed permanently: provider=%s, namespace=%s, "
            "attempts=%s, error=%s",
            self.provider_name,
            cache_namespace,
            self.max_retries,
            last_error,
        )
        if isinstance(last_error, ProviderError):
            raise last_error
        raise ProviderError(str(last_error) if last_error else "provider request failed")

    def _force_json_only_messages(
        self,
        messages: Sequence[dict[str, str]],
    ) -> list[dict[str, str]]:
        forced = [dict(message) for message in messages]
        if not forced:
            return forced
        forced[-1]["content"] = (
            forced[-1]["content"].rstrip()
            + "\n\nReturn only valid JSON with no markdown fences and no extra text."
        )
        return forced

    def _prompt_length_stats(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> tuple[int, int, dict[str, int]]:
        total_chars = 0
        total_bytes = 0
        role_chars: dict[str, int] = {}
        for message in messages:
            role = str(message.get("role", "unknown") or "unknown")
            content = str(message.get("content", "") or "")
            content_chars = len(content)
            total_chars += content_chars
            total_bytes += len(content.encode("utf-8"))
            role_chars[role] = role_chars.get(role, 0) + content_chars
        return total_chars, total_bytes, role_chars

    def _coerce_json_payload(self, content: Any) -> Any:
        if isinstance(content, (dict, list)):
            return content
        if isinstance(content, str):
            return extract_json_from_text(content)
        if isinstance(content, Sequence):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            if text_parts:
                return extract_json_from_text("\n".join(text_parts))
        raise ProviderError(
            f"unsupported provider response payload: {type(content)!r}"
        )

    def _extract_openai_chat_content(self, payload: Mapping[str, Any]) -> Any:
        try:
            choice = payload["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.exception(
                "%s response is missing choices[0].message",
                self.provider_name,
            )
            raise ProviderError(
                f"{self.provider_name} response does not contain choices[0].message"
            ) from exc
        if not isinstance(choice, Mapping) or not isinstance(message, Mapping):
            raise ProviderError(
                f"{self.provider_name} response choices[0].message has unexpected type: "
                f"{type(message)!r}"
            )

        content = message.get("content")
        if content is not None:
            return content

        finish_reason = choice.get("finish_reason")
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        refusal = message.get("refusal")
        message_keys = sorted(str(key) for key in message)
        logger.error(
            "%s response message.content is null: finish_reason=%s, "
            "message_keys=%s, reasoning_chars=%s, refusal_present=%s",
            self.provider_name,
            finish_reason,
            message_keys,
            len(str(reasoning)),
            refusal is not None,
        )
        raise ProviderError(
            f"{self.provider_name} response message.content is null "
            f"(finish_reason={finish_reason!r}, message_keys={message_keys}, "
            f"reasoning_chars={len(str(reasoning))}). "
            "For thinking models such as Qwen, try a larger --max-tokens value "
            "or pass --disable-thinking when using --provider vllm."
        )

    @abstractmethod
    def _generate_content(
        self,
        *,
        messages: Sequence[dict[str, str]],
        schema: dict[str, Any] | None,
    ) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError
