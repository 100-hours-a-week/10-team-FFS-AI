"""커머스 배치처리 데이터 모델."""

from pydantic import Field

from app.common.schemas import BaseSchema


class NaverProduct(BaseSchema):
    """네이버 쇼핑 API 응답에서 정제된 상품 데이터."""

    product_id: str = Field(..., description="네이버 상품 고유 ID")
    title: str = Field(..., description="상품명 (HTML 태그 제거)")
    link: str = Field(default="", description="상품 구매 링크")
    image_url: str = Field(default="", description="상품 이미지 URL")
    price: int = Field(default=0, description="최저가 (원)")
    mall_name: str = Field(default="", description="쇼핑몰명")
    brand: str | None = Field(default=None, description="브랜드")
    maker: str | None = Field(default=None, description="제조사")
    category1: str = Field(default="", description="대분류")
    category2: str | None = Field(default=None, description="중분류")
    category3: str | None = Field(default=None, description="소분류")
    category4: str | None = Field(default=None, description="세분류")
    search_keyword: str = Field(default="", description="검색에 사용된 키워드")


class BatchResult(BaseSchema):
    """배치 실행 결과."""

    total_fetched: int = Field(default=0, description="크롤링된 총 상품 수")
    total_skipped: int = Field(default=0, description="중복으로 스킵된 수")
    total_analyzed: int = Field(default=0, description="이미지 분석 완료 수")
    total_upserted: int = Field(default=0, description="Qdrant 저장 성공 수")
    total_failed: int = Field(default=0, description="처리 실패 수")
    duration_sec: float = Field(default=0.0, description="총 소요 시간 (초)")
