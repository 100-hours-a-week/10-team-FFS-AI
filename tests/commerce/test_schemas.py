"""커머스 모듈 스키마 유효성 검사 테스트."""

from app.commerce.schemas import BatchResult, NaverProduct

# ── NaverProduct 테스트 ──


def test_naver_product_basic() -> None:
    """필수 필드로 NaverProduct 생성."""
    p = NaverProduct(
        product_id="12345",
        title="테스트 상품",
        price=10000,
    )
    assert p.product_id == "12345"
    assert p.title == "테스트 상품"
    assert p.price == 10000
    assert p.link == ""
    assert p.image_url == ""
    assert p.brand is None
    assert p.maker is None
    assert p.search_keyword == ""


def test_naver_product_all_fields() -> None:
    """모든 필드가 올바르게 설정되는지 확인."""
    p = NaverProduct(
        product_id="99999",
        title="나이키 에어포스 1",
        link="https://example.com",
        image_url="https://img.example.com/shoe.jpg",
        price=139000,
        mall_name="무신사",
        brand="나이키",
        maker="Nike Inc",
        category1="패션의류",
        category2="남성신발",
        category3="스니커즈",
        category4=None,
        search_keyword="남성 스니커즈",
    )
    assert p.brand == "나이키"
    assert p.maker == "Nike Inc"
    assert p.category1 == "패션의류"
    assert p.category3 == "스니커즈"
    assert p.category4 is None


def test_naver_product_empty_brand_becomes_none() -> None:
    """빈 문자열 brand/maker는 None으로 처리 가능."""
    p = NaverProduct(
        product_id="111",
        title="무브랜드 상품",
        brand=None,
        maker=None,
    )
    assert p.brand is None
    assert p.maker is None


def test_naver_product_zero_price() -> None:
    """가격이 0인 상품도 허용."""
    p = NaverProduct(product_id="222", title="무료 상품", price=0)
    assert p.price == 0


# ── BatchResult 테스트 ──


def test_batch_result_defaults() -> None:
    """BatchResult 기본값 확인."""
    r = BatchResult()
    assert r.total_fetched == 0
    assert r.total_skipped == 0
    assert r.total_analyzed == 0
    assert r.total_upserted == 0
    assert r.total_failed == 0
    assert r.duration_sec == 0.0


def test_batch_result_custom_values() -> None:
    """BatchResult에 값을 설정."""
    r = BatchResult(
        total_fetched=100,
        total_skipped=80,
        total_analyzed=15,
        total_upserted=15,
        total_failed=5,
        duration_sec=120.5,
    )
    assert r.total_fetched == 100
    assert r.total_skipped == 80
    assert r.total_failed == 5
    assert r.duration_sec == 120.5
