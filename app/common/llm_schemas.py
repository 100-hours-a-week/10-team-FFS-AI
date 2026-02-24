from typing import Literal

from pydantic import BaseModel, Field

CategoryType = Literal["TOP", "BOTTOM", "DRESS", "SHOES", "ACCESSORY", "ETC"]


class ImageMajorAttributes(BaseModel):
    category: CategoryType = Field(description="의류 카테고리")
    color: list[str] = Field(default_factory=list, description="색상 목록")
    material: list[str] = Field(default_factory=list, description="소재 목록")
    style_tags: list[str] = Field(default_factory=list, description="스타일 태그 목록")


class ImageExtraMetadata(BaseModel):
    gender: str | None = Field(default=None, description="성별")
    season: list[str] = Field(default_factory=list, description="계절 목록")
    formality: str | None = Field(default=None, description="격식 수준")
    fit: str | None = Field(default=None, description="핏")
    occasion: list[str] = Field(default_factory=list, description="상황/장소 목록")


class ImageExtraAttributes(BaseModel):
    meta_data: ImageExtraMetadata = Field(default_factory=ImageExtraMetadata)
    caption: str | None = Field(default=None, description="이미지 설명")


class ImageAnalysisResult(BaseModel):
    major: ImageMajorAttributes
    extra: ImageExtraAttributes


# ---------------------------------------------------------------------------
# Outfit: Query Parser 응답 모델
# ---------------------------------------------------------------------------


class ReferenceItemLLM(BaseModel):
    category: CategoryType | None = Field(default=None, description="카테고리")
    color: str | None = Field(default=None, description="색상")
    style: str | None = Field(default=None, description="스타일")
    description: str | None = Field(default=None, description="기타 설명")


class OutfitQueryLLMResponse(BaseModel):
    occasion: str = Field(default="일상", description="상황/장소")
    style: str = Field(default="깔끔한", description="원하는 스타일")
    season: str | None = Field(default=None, description="계절")
    formality: str | None = Field(default=None, description="격식 수준")
    reference_item: ReferenceItemLLM | None = Field(
        default=None, description="기준 아이템"
    )
    target_category: CategoryType | None = Field(
        default=None, description="찾는 카테고리"
    )
    constraints: list[str] = Field(default_factory=list, description="추가 제약사항")


class OutfitItemLLM(BaseModel):
    clothes_id: int = Field(description="의류 ID")
    role: str = Field(default="기타", description="코디 내 역할 (상의, 하의 등)")


class OutfitCombination(BaseModel):
    description: str = Field(default="", description="코디 설명")
    items: list[OutfitItemLLM] = Field(default_factory=list, description="아이템 목록")


class OutfitCompositionLLMResponse(BaseModel):
    query_summary: str = Field(default="코디 추천", description="요청 한 줄 요약")
    outfits: list[OutfitCombination] = Field(
        default_factory=list, description="추천 코디 목록"
    )


class ShopQueryLLMResponse(BaseModel):
    occasion: str = Field(default="일상", description="상황/장소")
    style: str = Field(default="깔끔한", description="원하는 스타일")
    season: str | None = Field(default=None, description="계절")
    price_max: int | None = Field(default=None, description="최대 가격 (원)")
    price_min: int | None = Field(default=None, description="최소 가격 (원)")
    brand: str | None = Field(default=None, description="브랜드")
    target_category: CategoryType | None = Field(
        default=None, description="찾는 카테고리"
    )
    constraints: list[str] = Field(default_factory=list, description="추가 제약사항")


class ShopItemLLM(BaseModel):
    product_id: str = Field(description="상품 ID")


class ShopCombination(BaseModel):
    items: list[ShopItemLLM] = Field(default_factory=list, description="상품 목록")


class ShopCompositionLLMResponse(BaseModel):
    query_summary: str = Field(default="쇼핑 코디 추천", description="요청 한 줄 요약")
    outfits: list[ShopCombination] = Field(
        default_factory=list, description="추천 코디 목록"
    )
