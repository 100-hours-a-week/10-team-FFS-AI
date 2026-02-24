from pydantic import Field

from app.common.schemas import BaseSchema


class ShopParsedQuery(BaseSchema):
    occasion: str = Field(default="일상", description="상황/장소")
    style: str = Field(default="깔끔한", description="원하는 스타일")
    season: str | None = Field(default=None, description="계절")
    price_max: int | None = Field(default=None, description="최대 가격 (원)")
    price_min: int | None = Field(default=None, description="최소 가격 (원)")
    brand: str | None = Field(default=None, description="브랜드 필터")
    target_category: str | None = Field(
        default=None, description="찾는 아이템 카테고리"
    )
    constraints: list[str] = Field(default_factory=list, description="추가 제약사항")

    def is_full_outfit_request(self: "ShopParsedQuery") -> bool:
        return self.target_category is None


class ShopSearchQuery(BaseSchema):
    text: str = Field(..., description="임베딩할 텍스트")
    category_filter: str | None = Field(default=None, description="카테고리 필터")


class ProductCandidate(BaseSchema):
    product_id: str = Field(..., description="상품 고유 ID")
    title: str = Field(..., description="상품명")
    brand: str = Field(default="", description="브랜드")
    price: int = Field(default=0, description="가격 (원)")
    image_url: str = Field(default="", description="상품 이미지 URL")
    link: str = Field(default="", description="상품 구매 링크")
    source: str = Field(default="", description="쇼핑몰 출처")
    category: str = Field(default="", description="카테고리")
    style_tags: list[str] = Field(default_factory=list, description="스타일 태그")
    similarity_score: float = Field(..., description="유사도 점수 (0~1)")


class ProductSearchResult(BaseSchema):
    category: str = Field(..., description="검색 카테고리")
    candidates: list[ProductCandidate] = Field(
        default_factory=list, description="상품 후보 목록"
    )


class ShopSearchRequest(BaseSchema):
    user_id: int = Field(..., description="사용자 ID")
    query: str = Field(..., description="자연어 검색 쿼리")
    session_id: str | None = Field(default=None, description="멀티턴 대화 세션 ID")


class ShopProduct(BaseSchema):
    product_id: str = Field(..., description="상품 고유 ID")
    title: str = Field(..., description="상품명")
    brand: str = Field(default="", description="브랜드")
    price: int = Field(default=0, description="가격 (원)")
    image_url: str = Field(default="", description="상품 이미지 URL")
    link: str = Field(default="", description="상품 구매 링크")
    source: str = Field(default="", description="쇼핑몰 출처")
    category: str = Field(default="", description="카테고리")


class ShopOutfit(BaseSchema):
    outfit_id: str = Field(..., description="코디 고유 ID")
    items: list[ShopProduct] = Field(
        default_factory=list, description="코디 구성 상품 목록"
    )


class ShopSearchResponse(BaseSchema):
    query_summary: str = Field(..., description="검색 요약")
    outfits: list[ShopOutfit] = Field(
        default_factory=list, description="추천 코디 목록"
    )
    session_id: str | None = Field(default=None, description="멀티턴 대화 세션 ID")
