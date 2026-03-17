from typing import Literal

from pydantic import BaseModel, Field

CategoryType = Literal["TOP", "BOTTOM", "OUTER", "DRESS", "SHOES", "ACCESSORY", "ETC"]


SubCategoryType = Literal[
    # TOP
    "반소매_티셔츠",
    "긴소매_티셔츠",
    "셔츠_블라우스",
    "맨투맨_스웨트",
    "니트_스웨터",
    # BOTTOM
    "데님_팬츠",
    "슬랙스_트라우저",
    "트레이닝_조거",
    "숏츠",
    "스커트",
    "레깅스",
    # OUTER
    "가디건_니트아우터",
    "집업_후드아우터",
    "자켓",
    "코트",
    "패딩",
    "블레이저_수트자켓",
    # SHOES
    "스니커즈",
    "로퍼_단화",
    "구두_힐",
    "부츠",
    "샌들_슬리퍼",
    # DRESS
    "원피스",
    "점프슈트",
    # ACCESSORY
    "모자",
    "스카프_넥",
    "주얼리",
    "벨트",
    "양말_레그웨어",
    "백팩",
    "크로스_숄더백",
    "클러치_파우치",
    "웨이스트백",
]


CATEGORY_DETAIL: dict[str, list[str]] = {
    "TOP": [
        "반소매_티셔츠",
        "긴소매_티셔츠",
        "셔츠_블라우스",
        "맨투맨_스웨트",
        "니트_스웨터",
    ],
    "BOTTOM": [
        "데님_팬츠",
        "슬랙스_트라우저",
        "트레이닝_조거",
        "숏츠",
        "스커트",
        "레깅스",
    ],
    "OUTER": [
        "가디건_니트아우터",
        "집업_후드아우터",
        "자켓",
        "코트",
        "패딩",
        "블레이저_수트자켓",
    ],
    "SHOES": ["스니커즈", "로퍼_단화", "구두_힐", "부츠", "샌들_슬리퍼"],
    "DRESS": ["원피스", "점프슈트"],
    "ACCESSORY": [
        "모자",
        "스카프_넥",
        "주얼리",
        "벨트",
        "양말_레그웨어",
        "백팩",
        "크로스_숄더백",
        "클러치_파우치",
        "웨이스트백",
    ],
    "ETC": [],
}


class ImageMajorAttributes(BaseModel):
    category: CategoryType = Field(description="의류 카테고리")
    sub_category: SubCategoryType | None = Field(
        default=None, description="세부 카테고리 (L2) — 임베딩/payload 전용"
    )
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
    sub_categories: dict[str, list[str]] | None = Field(
        default=None,
        description="카테고리별 세부 유형 (예: {'TOP': ['맨투맨_스웨트'], 'OUTER': ['패딩']})",
    )


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


class IntentDetectionResult(BaseModel):
    """멀티턴 대화 인텐트 판별 결과"""

    intent: Literal["new_outfit", "modify_previous", "confirm_item"] = Field(
        ...,
        description=(
            "사용자 의도 - new_outfit: 새로운 코디 추천, "
            "modify_previous: 이전 추천 수정, "
            "confirm_item: 아이템 확정"
        ),
    )
