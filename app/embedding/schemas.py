from typing import List

from pydantic import BaseModel, Field


class ClothingMetadata(BaseModel):
    category: str = Field(..., description="의류 카테고리 (예: 상의, 하의)")
    color: List[str] = Field(..., description="의류 색상 목록")
    material: List[str] = Field(..., description="의류 소재 목록")


class EmbeddingRequest(BaseModel):
    userId: str = Field(..., description="사용자 ID")
    clothesId: int = Field(..., description="의류 ID")
    imageUrl: str = Field(..., description="의류 이미지 URL")
    metadata: ClothingMetadata = Field(..., description="의류 메타데이터")


class EmbeddingResponse(BaseModel):
    clothesId: int = Field(..., description="의류 ID")
    caption: str = Field(..., description="생성된 캡션 텍스트")
    indexed: bool = Field(..., description="벡터 DB 저장 여부")


class DeleteResponse(BaseModel):
    clothesId: int = Field(..., description="의류 ID")
    deleted: bool = Field(..., description="삭제 여부")
