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
