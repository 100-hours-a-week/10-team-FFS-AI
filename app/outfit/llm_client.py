import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx
from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings, get_settings
from app.outfit.exceptions import LLMError

logger = logging.getLogger(__name__)


def _is_retryable_exception(exception: BaseException) -> bool:
    if isinstance(exception, httpx.TimeoutException | httpx.ConnectError):
        return True

    if isinstance(exception, httpx.HTTPStatusError):
        status = exception.response.status_code
        # 429 (Rate Limit), 5xx (Server Error)만 재시도
        return status == 429 or status >= 500

    return False


def _wait_with_retry_after(retry_state: RetryCallState) -> float:
    exception = retry_state.outcome.exception()

    if isinstance(exception, httpx.HTTPStatusError):
        if exception.response.status_code == 429:
            retry_after = exception.response.headers.get("Retry-After")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass  # 파싱 실패 시 기본 대기

    # 지수 백오프
    return wait_exponential(multiplier=1, min=2, max=30)(retry_state)


class LLMClient(ABC):
    @abstractmethod
    async def chat_completion(
        self: "LLMClient",
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        ...


class OpenAIClient(LLMClient):
    def __init__(self: "OpenAIClient", settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.api_key = self.settings.openai_api_key
        self.model = self.settings.openai_chat_model
        self.base_url = "https://api.openai.com/v1"

    def _get_headers(self: "OpenAIClient") -> dict[str, str]:
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not configured")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self: "OpenAIClient",
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=_wait_with_retry_after,
        retry=retry_if_exception(_is_retryable_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _request(self: "OpenAIClient", payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()

        async with httpx.AsyncClient(timeout=self.settings.llm_timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

    async def chat_completion(
        self: "OpenAIClient",
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        payload = self._build_payload(messages, temperature, max_tokens)

        try:
            return await self._request(payload)

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            logger.error("OpenAI API Error [%s]: %s", status, e.response.text)

            if status == 401:
                raise LLMError("Invalid OpenAI API Key") from e
            if status == 400:
                raise LLMError(f"Invalid request: {e.response.text}") from e
            # 429, 5xx는 재시도 후에도 실패한 경우
            raise LLMError(f"OpenAI API failed after retries: {e.response.text}") from e

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.error("OpenAI network error after retries: %s", e)
            raise LLMError(f"Network error: {e}") from e

        except Exception as e:
            logger.exception("Unexpected error during OpenAI call")
            raise LLMError(f"Unexpected error: {e}") from e
