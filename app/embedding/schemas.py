from pydantic import Field

from app.common.schemas import BaseSchema


class ClothingMetadata(BaseSchema):
    category: str = Field(..., description="의류 카테고리 (예: 상의, 하의)")
    color: list[str] = Field(..., description="의류 색상 목록")
    material: list[str] = Field(..., description="의류 소재 목록")
    style_tags: list[str] = Field(default=[], description="스타일 태그 목록")
    gender: str = Field(..., description="성별 (예: 남성, 여성, 공용)")
    season: list[str] = Field(..., description="추천 계절 목록")
    formality: str = Field(..., description="포멀도 (예: 포멀, 캐주얼)")
    fit: str = Field(..., description="핏 (예: 오버핏, 슬림핏)")
    occasion: list[str] = Field(default=[], description="적합한 상황/장소 (예: 면접, 출근, 데이트)")


class EmbeddingRequest(BaseSchema):
    user_id: str = Field(..., description="사용자 ID")
    clothes_id: int = Field(..., description="의류 ID")
    image_url: str = Field(..., description="의류 이미지 URL")
    caption: str = Field(..., description="이미지 분석 단계에서 생성된 캡션")
    metadata: ClothingMetadata = Field(..., description="의류 확장 메타데이터")


class EmbeddingResponse(BaseSchema):
    clothes_id: int = Field(..., description="의류 ID")
    caption: str = Field(..., description="생성된 캡션 텍스트")
    indexed: bool = Field(..., description="벡터 DB 저장 여부")


class DeleteResponse(BaseSchema):
    clothes_id: int = Field(..., description="의류 ID")
    deleted: bool = Field(..., description="삭제 여부")
