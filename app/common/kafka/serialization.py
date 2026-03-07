import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class DeserializationError(Exception):
    def __init__(self, message: str, original_data: bytes) -> None:
        super().__init__(message)
        self.original_data = original_data


def serialize(model: BaseModel) -> bytes:
    return model.model_dump_json(by_alias=True).encode("utf-8")


def deserialize(data: bytes, model_class: type[T]) -> T:
    try:
        decoded_data = data.decode("utf-8")
        json_dict = json.loads(decoded_data)

        return model_class.model_validate(json_dict)

    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise DeserializationError(f"Invalid JSON format: {str(e)}", data) from e

    except ValidationError as e:
        raise DeserializationError(f"Schema validation failed: {str(e)}", data) from e
