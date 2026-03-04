from pydantic import Field

from app.common.llm_schemas import CategoryType
from app.common.schemas import BaseSchema


class ReferenceItem(BaseSchema):
    category: CategoryType | None = Field(
        default=None, description="카테고리 (TOP, BOTTOM, DRESS, SHOES, ACCESSORY, ETC)"
    )
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
    target_category: CategoryType | None = Field(
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


class Weather(BaseSchema):
    temperature: int = Field(..., description="온도 (섭씨)")
    condition: str = Field(..., description="날씨 상태 (sunny, cloudy, rainy 등)")


class UploadSlot(BaseSchema):
    file_id: int = Field(..., description="파일 ID")
    object_key: str = Field(..., description="S3에 저장할 이미지 파일의 키")
    presigned_url: str = Field(..., description="S3에 저장할 이미지 파일의 URL")


class OutfitRequest(BaseSchema):
    user_id: int = Field(..., description="사용자 ID")
    query: str = Field(..., description="사용자 자연어 요청")
    session_id: str | None = Field(default=None, description="멀티턴 대화 세션 ID")
    weather: Weather | None = Field(default=None, description="날씨 정보")
    urls: list[UploadSlot] = Field(
        default_factory=list, description="VTON용 업로드 슬롯 정보"
    )


class ClothingCandidate(BaseSchema):
    clothes_id: int = Field(..., description="의류 ID")
    image_url: str = Field(..., description="이미지 URL")
    category: CategoryType = Field(
        ..., description="카테고리 (TOP, BOTTOM, DRESS, SHOES, ACCESSORY, ETC)"
    )
    color: list[str] = Field(default_factory=list, description="색상 목록")
    style_tags: list[str] = Field(default_factory=list, description="스타일 태그")
    caption: str | None = Field(default=None, description="캡션")
    material: list[str] = Field(default_factory=list, description="소재")
    season: list[str] = Field(default_factory=list, description="계절")
    formality: str = Field(default="", description="포멀/캐주얼 등급")
    occasion: list[str] = Field(default_factory=list, description="착용 상황")
    similarity_score: float = Field(..., description="유사도 점수 (0~1)")

    source: str = Field(default="closet", description="아이템 출처: 'closet' | 'shop'")

    price: int | None = Field(default=None, description="가격 (원)")
    brand: str | None = Field(default=None, description="브랜드")
    purchase_url: str | None = Field(default=None, description="구매 링크")


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
    clothes_ids: list[int] = Field(default_factory=list, description="의류 ID 목록")
    items: list[OutfitItem] = Field(
        default_factory=list, description="아이템 상세 정보", exclude=True
    )
    fallback_notice: str | None = Field(
        default=None, description="대체 안내 메시지 (아이템 부족 시)"
    )
    file_id: int | None = Field(
        default=None, description="추천한 코디 조합 착용한 이미지 파일 ID"
    )
    vton_error: str | None = Field(
        default=None, description="VTON 이미지 생성 실패 시 에러 메시지", exclude=True
    )


class OutfitResponse(BaseSchema):
    query_summary: str = Field(..., description="사용자 요청 요약")
    outfits: list[Outfit] = Field(default_factory=list, description="추천 코디 목록")
    session_id: str | None = Field(default=None, description="멀티턴 대화 세션 ID")
