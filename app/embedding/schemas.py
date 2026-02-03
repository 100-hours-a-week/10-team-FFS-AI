from pydantic import Field

from app.common.schemas import BaseSchema


class EmbeddingResponse(BaseSchema):
    clothes_id: int = Field(..., description="의류 ID")
    caption: str = Field(..., description="생성된 캡션 텍스트")
    indexed: bool = Field(..., description="벡터 DB 저장 여부")


class DeleteResponse(BaseSchema):
    clothes_id: int = Field(..., description="의류 ID")
    deleted: bool = Field(..., description="삭제 여부")
