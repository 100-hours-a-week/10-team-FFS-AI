from pydantic import Field

from app.common.schemas import BaseSchema


class ReferenceItem(BaseSchema):
    category: str | None = Field(default=None, description="카테고리 (코트, 셔츠 등)")
    color: str | None = Field(default=None, description="색상")
    style: str | None = Field(default=None, description="스타일 (오버핏, 캐주얼 등)")
    description: str | None = Field(default=None, description="기타 설명")


class ParsedQuery(BaseSchema):
    occasion: str = Field(default="일상", description="상황/장소 (면접, 데이트, 출근)")
    style: str = Field(default="깔끔한", description="원하는 스타일")
    season: str | None = Field(default=None, description="계절")
    formality: str | None = Field(
        default=None, description="격식 수준 (포멀, 캐주얼, 세미포멀)"
    )

    reference_item: ReferenceItem | None = Field(
        default=None, description="기준 아이템"
    )
    target_category: str | None = Field(
        default=None, description="찾는 아이템 카테고리"
    )

    constraints: list[str] = Field(default_factory=list, description="추가 제약사항")

    def is_full_outfit_request(self: "ParsedQuery") -> bool:
        return self.target_category is None

    def is_matching_request(self: "ParsedQuery") -> bool:
        return self.reference_item is not None


class SearchQuery(BaseSchema):
    text: str = Field(..., description="임베딩할 텍스트")
    category_filter: str | None = Field(default=None, description="카테고리 필터")


class OutfitRequest(BaseSchema):
    user_id: str = Field(..., description="사용자 ID")
    query: str = Field(..., description="사용자 자연어 요청")


class ClothingCandidate(BaseSchema):
    clothes_id: int = Field(..., description="의류 ID")
    image_url: str = Field(..., description="이미지 URL")
    category: str = Field(..., description="카테고리 (상의, 하의, 아우터 등)")
    color: list[str] = Field(default_factory=list, description="색상 목록")
    style_tags: list[str] = Field(default_factory=list, description="스타일 태그")
    caption: str | None = Field(default=None, description="캡션")
    similarity_score: float = Field(..., description="유사도 점수 (0~1)")


class SearchResult(BaseSchema):
    category: str = Field(..., description="검색 카테고리")
    candidates: list[ClothingCandidate] = Field(
        default_factory=list, description="후보 아이템 목록"
    )


class OutfitItem(BaseSchema):
    clothes_id: int = Field(..., description="의류 ID")
    image_url: str = Field(..., description="이미지 URL")
    category: str = Field(..., description="카테고리")
    role: str = Field(..., description="코디 내 역할 (상의, 하의 등)")


class Outfit(BaseSchema):
    outfit_id: str = Field(..., description="코디 고유 ID (UUID)")
    description: str = Field(..., description="코디 설명")
    items: list[OutfitItem] = Field(default_factory=list, description="아이템 목록")
    fallback_notice: str | None = Field(
        default=None, description="대체 안내 메시지 (아이템 부족 시)"
    )


class OutfitResponse(BaseSchema):
    query_summary: str = Field(..., description="사용자 요청 요약")
    outfits: list[Outfit] = Field(default_factory=list, description="추천 코디 목록")
