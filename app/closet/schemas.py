from pydantic import Field

from app.common.llm_schemas import CategoryType
from app.common.schemas import BaseSchema


class MajorAttributes(BaseSchema):
    category: CategoryType = Field(..., description="카테고리")
    color: list[str] = Field(default_factory=list, description="색상 목록")
    material: list[str] = Field(default_factory=list, description="소재 목록")
    style_tags: list[str] = Field(default_factory=list, description="스타일 태그 목록")


class ExtraMetadata(BaseSchema):
    gender: str | None = Field(default=None, description="성별")
    season: list[str] = Field(default_factory=list, description="계절 목록")
    formality: str | None = Field(default=None, description="격식 수준")
    fit: str | None = Field(default=None, description="핏 (오버핏, 슬림핏 등)")
    occasion: list[str] = Field(default_factory=list, description="상황/장소 목록")


class ExtraAttributes(BaseSchema):
    meta_data: ExtraMetadata = Field(
        default_factory=ExtraMetadata, description="추가 메타데이터"
    )
    caption: str | None = Field(default=None, description="이미지 설명")


class EmbeddingRequest(BaseSchema):
    user_id: int = Field(..., description="사용자 ID")
    clothes_id: int = Field(..., description="의류 ID")
    image_url: str = Field(..., description="이미지 URL")
    major: MajorAttributes = Field(..., description="주요 속성")
    extra: ExtraAttributes = Field(..., description="추가 속성")
