from app.shop.schemas import (
    ProductCandidate,
    ProductSearchResult,
    ShopOutfit,
    ShopParsedQuery,
    ShopProduct,
    ShopSearchQuery,
    ShopSearchRequest,
    ShopSearchResponse,
)


class TestShopSearchRequest:
    def test_basic_request(self) -> None:
        request = ShopSearchRequest(
            user_id=123,
            query="Y2K 크롭탑 코디",
        )
        assert request.user_id == 123
        assert request.query == "Y2K 크롭탑 코디"
        assert request.session_id is None

    def test_request_with_session(self) -> None:
        request = ShopSearchRequest(
            user_id=123,
            query="캐주얼 코디",
            session_id="sess_001",
        )
        assert request.session_id == "sess_001"

    def test_camel_case_serialization(self) -> None:
        """BaseSchema의 camelCase alias 변환 확인"""
        request = ShopSearchRequest(
            user_id=123,
            query="테스트",
        )
        data = request.model_dump(by_alias=True)
        assert "userId" in data
        assert "sessionId" in data
        assert data["userId"] == 123

    def test_from_camel_case(self) -> None:
        """camelCase 입력에서 파싱"""
        request = ShopSearchRequest.model_validate(
            {
                "userId": 456,
                "query": "테스트 쿼리",
                "sessionId": "sess_002",
            }
        )
        assert request.user_id == 456
        assert request.session_id == "sess_002"


class TestShopParsedQuery:
    def test_defaults(self) -> None:
        parsed = ShopParsedQuery()
        assert parsed.occasion == "일상"
        assert parsed.style == "깔끔한"
        assert parsed.season is None
        assert parsed.price_max is None
        assert parsed.price_min is None
        assert parsed.brand is None
        assert parsed.target_category is None
        assert parsed.constraints == []

    def test_full_outfit_request(self) -> None:
        """target_category가 없으면 전체 코디 요청"""
        parsed = ShopParsedQuery(style="Y2K")
        assert parsed.is_full_outfit_request() is True

    def test_single_category_request(self) -> None:
        """target_category가 있으면 특정 카테고리 요청"""
        parsed = ShopParsedQuery(style="Y2K", target_category="TOP")
        assert parsed.is_full_outfit_request() is False


class TestProductCandidate:
    def test_minimal_candidate(self) -> None:
        candidate = ProductCandidate(
            product_id="prod_001",
            title="크롭탑",
            similarity_score=0.95,
        )
        assert candidate.product_id == "prod_001"
        assert candidate.brand == ""
        assert candidate.price == 0
        assert candidate.style_tags == []

    def test_full_candidate(self) -> None:
        candidate = ProductCandidate(
            product_id="prod_001",
            title="Y2K 크롭탑",
            brand="무신사스탠다드",
            price=29000,
            image_url="https://img.com/001.jpg",
            link="https://musinsa.com/001",
            source="musinsa",
            category="TOP",
            style_tags=["Y2K", "크롭"],
            similarity_score=0.92,
        )
        assert candidate.brand == "무신사스탠다드"
        assert candidate.price == 29000
        assert len(candidate.style_tags) == 2


class TestShopSearchResponse:
    def test_empty_response(self) -> None:
        response = ShopSearchResponse(
            query_summary="검색 결과가 없습니다",
            outfits=[],
        )
        assert response.outfits == []
        assert response.session_id is None

    def test_response_with_outfits(self) -> None:
        response = ShopSearchResponse(
            query_summary="Y2K 코디",
            outfits=[
                ShopOutfit(
                    outfit_id="outfit_s001",
                    items=[
                        ShopProduct(
                            product_id="prod_001",
                            title="크롭탑",
                            price=29000,
                        ),
                    ],
                ),
            ],
        )
        assert len(response.outfits) == 1
        assert len(response.outfits[0].items) == 1
        assert response.outfits[0].items[0].price == 29000

    def test_response_camel_case(self) -> None:
        """응답 직렬화 시 camelCase 확인"""
        response = ShopSearchResponse(
            query_summary="테스트",
            outfits=[
                ShopOutfit(
                    outfit_id="outfit_001",
                    items=[
                        ShopProduct(
                            product_id="prod_001",
                            title="상품명",
                            image_url="https://img.com/1.jpg",
                        ),
                    ],
                ),
            ],
        )
        data = response.model_dump(by_alias=True)
        assert "querySummary" in data
        assert "outfitId" in data["outfits"][0]
        assert "productId" in data["outfits"][0]["items"][0]
        assert "imageUrl" in data["outfits"][0]["items"][0]


class TestProductSearchResult:
    def test_empty_result(self) -> None:
        result = ProductSearchResult(category="TOP", candidates=[])
        assert result.category == "TOP"
        assert result.candidates == []

    def test_with_candidates(self) -> None:
        result = ProductSearchResult(
            category="TOP",
            candidates=[
                ProductCandidate(
                    product_id="p1",
                    title="셔츠",
                    similarity_score=0.9,
                ),
                ProductCandidate(
                    product_id="p2",
                    title="티셔츠",
                    similarity_score=0.85,
                ),
            ],
        )
        assert len(result.candidates) == 2


class TestShopSearchQuery:
    def test_basic_query(self) -> None:
        query = ShopSearchQuery(text="Y2K 스타일 TOP")
        assert query.text == "Y2K 스타일 TOP"
        assert query.category_filter is None

    def test_with_category(self) -> None:
        query = ShopSearchQuery(
            text="캐주얼 하의",
            category_filter="BOTTOM",
        )
        assert query.category_filter == "BOTTOM"
