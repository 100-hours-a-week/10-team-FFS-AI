

from __future__ import annotations

import json
import logging
from typing import TypeVar

from pydantic import ValidationError

from app.common.schemas import BaseSchema

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseSchema)


class DeserializationError(Exception):


    def __init__(
        self,
        message: str,
        original_data: bytes,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.original_data = original_data
        self.cause = cause

    def __str__(self) -> str:
        cause_info = f" (원인: {self.cause})" if self.cause else ""
        return f"{self.args[0]}{cause_info}"


def serialize(model: BaseSchema) -> bytes:

    return model.model_dump_json(by_alias=True).encode("utf-8")


def deserialize(data: bytes, model_class: type[T]) -> T:

    try:
        json_dict = json.loads(data.decode("utf-8"))
        return model_class.model_validate(json_dict)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 실패: {e}, 원본: {data[:200]}")
        raise DeserializationError(
            message=f"JSON 파싱 실패: {e}",
            original_data=data,
            cause=e,
        ) from e
    except ValidationError as e:
        logger.error(f"스키마 검증 실패: {e}, 원본: {data[:200]}")
        raise DeserializationError(
            message=f"스키마 검증 실패: {e.error_count()}개 오류",
            original_data=data,
            cause=e,
        ) from e
