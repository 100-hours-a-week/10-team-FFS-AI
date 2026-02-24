import logging
from abc import ABC, abstractmethod
from typing import Any

import openai
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.outfit.exceptions import LLMError

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    @abstractmethod
    async def chat_completion(
        self: "LLMClient",
        messages: list[dict[str, Any]],
        response_format: type | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> BaseModel | dict[str, Any]: ...


class OpenAIClient(LLMClient):
    def __init__(self: "OpenAIClient", settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

        if not self.settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not configured")

        self._client = openai.AsyncOpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url or None,
            timeout=float(self.settings.llm_timeout),
            max_retries=self.settings.llm_max_retries,
        )
        self.model = self.settings.openai_chat_model

    async def chat_completion(
        self: "OpenAIClient",
        messages: list[dict[str, Any]],
        response_format: type | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> BaseModel | dict[str, Any]:
        try:
            if response_format is not None:
                completion = await self._client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                parsed = completion.choices[0].message.parsed
                if parsed is None:
                    raise LLMError("Structured Output parsing returned None")
                return parsed

            else:
                completion = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return completion.model_dump()

        except openai.AuthenticationError as e:
            logger.error("OpenAI 인증 실패: %s", e)
            raise LLMError("Invalid OpenAI API Key") from e

        except openai.BadRequestError as e:
            logger.error("OpenAI 잘못된 요청 [400]: %s", e)
            raise LLMError(f"Invalid request: {e}") from e

        except openai.RateLimitError as e:
            logger.error("OpenAI 요청 한도 초과 [429]: %s", e)
            raise LLMError(f"OpenAI API failed after retries: {e}") from e

        except openai.APIStatusError as e:
            logger.error("OpenAI API 에러 [%s]: %s", e.status_code, e)
            raise LLMError(f"OpenAI API failed after retries: {e}") from e

        except openai.APIConnectionError as e:
            logger.error("OpenAI 네트워크 에러: %s", e)
            raise LLMError(f"Network error: {e}") from e

        except LLMError:
            raise

        except Exception as e:
            logger.exception("OpenAI 호출 중 예상치 못한 에러")
            raise LLMError(f"Unexpected error: {e}") from e
