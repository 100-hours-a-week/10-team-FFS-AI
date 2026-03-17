"""커머스 배치 서비스 테스트.

외부 의존성(vLLM, Redis, Qdrant, 네이버 API)을 mock하여 서비스 로직을 검증한다.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.closet.schemas import ExtraAttributes, ExtraMetadata, MajorAttributes
from app.commerce.schemas import NaverProduct
from app.commerce.service import CommerceBatchService


@pytest.fixture
def mock_settings() -> Generator[MagicMock, Any, None]:
    with patch("app.commerce.service.get_settings") as mock:
        settings = MagicMock()
        settings.vllm_server_url = "http://localhost:8001"
        settings.batch_start_hour = 2
        settings.batch_end_hour = 6
        settings.batch_max_products_per_keyword = 10
        mock.return_value = settings
        yield settings


# ── is_vllm_available 테스트 ──


@pytest.mark.asyncio
async def test_vllm_available_success(mock_settings: MagicMock) -> None:
    """vLLM 헬스체크 성공."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("app.commerce.service.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        service = CommerceBatchService()
        assert await service.is_vllm_available() is True


@pytest.mark.asyncio
async def test_vllm_available_failure(mock_settings: MagicMock) -> None:
    """vLLM 서버 연결 실패 → False."""
    with patch("app.commerce.service.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        service = CommerceBatchService()
        assert await service.is_vllm_available() is False


@pytest.mark.asyncio
async def test_vllm_available_no_url(mock_settings: MagicMock) -> None:
    """VLLM_SERVER_URL이 비어있으면 False."""
    mock_settings.vllm_server_url = ""
    service = CommerceBatchService()
    assert await service.is_vllm_available() is False


# ── _is_within_window 테스트 ──


def test_is_within_window_inside(mock_settings: MagicMock) -> None:
    """새벽 3시 → 윈도우 내."""
    from datetime import datetime, timedelta, timezone

    kst = timezone(timedelta(hours=9))

    service = CommerceBatchService()
    with patch("app.commerce.service.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 3, 15, 3, 0, tzinfo=kst)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        assert service._is_within_window() is True


def test_is_within_window_outside(mock_settings: MagicMock) -> None:
    """오전 7시 → 윈도우 밖."""
    from datetime import datetime, timedelta, timezone

    kst = timezone(timedelta(hours=9))

    service = CommerceBatchService()
    with patch("app.commerce.service.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 3, 15, 7, 0, tzinfo=kst)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        assert service._is_within_window() is False


# ── _process_product 테스트 ──


def _make_product(**kwargs: str | int | None) -> NaverProduct:
    defaults = {
        "product_id": "12345",
        "title": "테스트 상품",
        "link": "https://example.com",
        "image_url": "https://img.example.com/1.jpg",
        "price": 10000,
        "mall_name": "테스트몰",
        "category1": "패션의류",
        "search_keyword": "테스트",
    }
    defaults.update(kwargs)
    return NaverProduct(**defaults)


@pytest.mark.asyncio
async def test_process_product_skip(mock_settings: MagicMock) -> None:
    """기존 상품 + 변동 없음 → skipped."""
    service = CommerceBatchService()
    service.repository = MagicMock()
    service.repository.get_existing_product = AsyncMock(
        return_value={"price": 10000, "title": "테스트 상품"}
    )
    service.repository.is_product_changed = MagicMock(return_value=False)

    product = _make_product(price=10000, title="테스트 상품")
    mock_analyzer = MagicMock()

    result = await service._process_product(product, "TOP", mock_analyzer)
    assert result == "skipped"


@pytest.mark.asyncio
async def test_process_product_updated(mock_settings: MagicMock) -> None:
    """기존 상품 + 가격 변동 → updated."""
    service = CommerceBatchService()
    service.repository = MagicMock()
    service.repository.get_existing_product = AsyncMock(
        return_value={"price": 10000, "title": "테스트 상품"}
    )
    service.repository.is_product_changed = MagicMock(return_value=True)
    service.repository.update_payload = AsyncMock()

    product = _make_product(price=15000, title="테스트 상품")
    mock_analyzer = MagicMock()

    result = await service._process_product(product, "TOP", mock_analyzer)
    assert result == "updated"
    service.repository.update_payload.assert_called_once()


@pytest.mark.asyncio
async def test_process_product_new(mock_settings: MagicMock) -> None:
    """신규 상품 → new (분석 + 저장)."""
    service = CommerceBatchService()
    service.repository = MagicMock()
    service.repository.get_existing_product = AsyncMock(return_value=None)
    service.repository.upsert = AsyncMock(return_value=True)

    mock_analyzer = MagicMock()
    mock_analysis = MagicMock()
    mock_analysis.major.category = "TOP"
    mock_analysis.major.color = ["흰색"]
    mock_analysis.major.material = ["면"]
    mock_analysis.major.style_tags = ["캐주얼"]
    mock_analysis.extra.meta_data.gender = "남성"
    mock_analysis.extra.meta_data.season = ["봄"]
    mock_analysis.extra.meta_data.formality = "캐주얼"
    mock_analysis.extra.meta_data.fit = "레귤러"
    mock_analysis.extra.meta_data.occasion = ["일상"]
    mock_analysis.extra.caption = "흰색 면 티셔츠"
    mock_analyzer.analyze_image = AsyncMock(return_value=mock_analysis)

    service.formatter = MagicMock()
    service.formatter.format = MagicMock(return_value="embedding text")
    service.embedding_client = MagicMock()
    service.embedding_client.embed = AsyncMock(return_value=[0.1] * 4096)

    with patch.object(
        CommerceBatchService,
        "_download_image",
        new=AsyncMock(return_value=b"fake_image"),
    ):
        product = _make_product()
        result = await service._process_product(product, "TOP", mock_analyzer)

    assert result == "new"
    service.repository.upsert.assert_called_once()


# ── _build_payload 테스트 ──


def test_build_payload_structure() -> None:
    """payload가 shop 모듈 호환 구조인지 확인."""
    product = _make_product(brand="나이키")
    major = MajorAttributes(
        category="SHOES",
        color=["흰색"],
        material=["가죽"],
        style_tags=["캐주얼"],
    )
    extra = ExtraAttributes(
        meta_data=ExtraMetadata(
            gender="남성",
            season=["봄", "가을"],
            formality="캐주얼",
            fit="레귤러",
            occasion=["일상"],
        ),
        caption="흰색 가죽 스니커즈",
    )

    payload = CommerceBatchService._build_payload(
        product, major, extra, "embedding text"
    )

    # shop/repository.py의 _to_candidate()가 읽는 필드 확인
    assert payload["productId"] == "12345"
    assert payload["title"] == "테스트 상품"
    assert payload["price"] == 10000
    assert payload["brand"] == "나이키"
    assert payload["source"] == "naver"
    assert payload["category"] == "SHOES"  # 소문자 c
    assert payload["color"] == ["흰색"]
    assert payload["material"] == ["가죽"]
    assert payload["styleTags"] == ["캐주얼"]
    assert payload["season"] == ["봄", "가을"]
    assert payload["formality"] == "캐주얼"
    assert payload["occasion"] == ["일상"]
    assert payload["embeddingText"] == "embedding text"


def test_build_payload_brand_none() -> None:
    """brand가 None이면 빈 문자열로 저장."""
    product = _make_product(brand=None)
    major = MajorAttributes(category="TOP", color=[], material=[], style_tags=[])
    extra = ExtraAttributes(
        meta_data=ExtraMetadata(season=[], occasion=[]),
        caption="",
    )

    payload = CommerceBatchService._build_payload(product, major, extra, "")
    assert payload["brand"] == ""


# ── _download_image 테스트 ──


@pytest.mark.asyncio
async def test_download_image_success() -> None:
    """이미지 다운로드 성공."""
    with patch("app.commerce.service.httpx.AsyncClient") as mock_client_cls:
        mock_response = MagicMock()
        mock_response.content = b"fake_image_bytes"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        result = await CommerceBatchService._download_image("https://img.com/1.jpg")
        assert result == b"fake_image_bytes"
