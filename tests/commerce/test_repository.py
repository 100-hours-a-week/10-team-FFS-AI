"""커머스 Qdrant 저장소 테스트.

Qdrant 클라이언트를 mock하여 저장소 로직을 검증한다.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.commerce.repository import CommerceRepository, _product_id_to_point_id

# ── _product_id_to_point_id 테스트 ──


def test_product_id_numeric() -> None:
    """숫자 productId는 int로 변환."""
    assert _product_id_to_point_id("12345") == 12345


def test_product_id_large_numeric() -> None:
    """큰 숫자도 정상 변환."""
    assert _product_id_to_point_id("89403378798") == 89403378798


def test_product_id_non_numeric() -> None:
    """비숫자 productId는 hash로 변환."""
    result = _product_id_to_point_id("abc-xyz")
    assert isinstance(result, int)
    assert 0 <= result < 2**63


def test_product_id_deterministic() -> None:
    """같은 입력에 대해 항상 같은 결과."""
    assert _product_id_to_point_id("test") == _product_id_to_point_id("test")


# ── CommerceRepository 테스트 ──


@pytest.fixture
def mock_settings() -> Generator[MagicMock, Any, None]:
    with patch("app.commerce.repository.get_settings") as mock:
        settings = MagicMock()
        settings.qdrant_shop_collection_name = "commerce"
        mock.return_value = settings
        yield settings


@pytest.fixture
def mock_qdrant() -> AsyncMock:
    return AsyncMock()


@pytest.mark.asyncio
async def test_get_existing_product_found(
    mock_settings: MagicMock, mock_qdrant: AsyncMock
) -> None:
    """Qdrant에 존재하는 상품 조회."""
    mock_point = MagicMock()
    mock_point.payload = {"title": "나이키 신발", "price": 100000}
    mock_qdrant.retrieve = AsyncMock(return_value=[mock_point])

    repo = CommerceRepository(qdrant_client=mock_qdrant)
    result = await repo.get_existing_product("12345")

    assert result is not None
    assert result["title"] == "나이키 신발"
    assert result["price"] == 100000


@pytest.mark.asyncio
async def test_get_existing_product_not_found(
    mock_settings: MagicMock, mock_qdrant: AsyncMock
) -> None:
    """Qdrant에 없는 상품 조회 → None."""
    mock_qdrant.retrieve = AsyncMock(return_value=[])

    repo = CommerceRepository(qdrant_client=mock_qdrant)
    result = await repo.get_existing_product("99999")

    assert result is None


@pytest.mark.asyncio
async def test_get_existing_product_exception(
    mock_settings: MagicMock, mock_qdrant: AsyncMock
) -> None:
    """Qdrant 조회 실패 → None (에러 삼킴)."""
    mock_qdrant.retrieve = AsyncMock(side_effect=Exception("connection failed"))

    repo = CommerceRepository(qdrant_client=mock_qdrant)
    result = await repo.get_existing_product("12345")

    assert result is None


# ── is_product_changed 테스트 ──


def test_is_product_changed_no_change(
    mock_settings: MagicMock, mock_qdrant: AsyncMock
) -> None:
    """가격/제목 동일 → 변경 없음."""
    repo = CommerceRepository(qdrant_client=mock_qdrant)
    existing = {"price": 10000, "title": "상품A"}
    assert repo.is_product_changed(existing, 10000, "상품A") is False


def test_is_product_changed_price_changed(
    mock_settings: MagicMock, mock_qdrant: AsyncMock
) -> None:
    """가격 변경 → 변경됨."""
    repo = CommerceRepository(qdrant_client=mock_qdrant)
    existing = {"price": 10000, "title": "상품A"}
    assert repo.is_product_changed(existing, 15000, "상품A") is True


def test_is_product_changed_title_changed(
    mock_settings: MagicMock, mock_qdrant: AsyncMock
) -> None:
    """제목 변경 → 변경됨."""
    repo = CommerceRepository(qdrant_client=mock_qdrant)
    existing = {"price": 10000, "title": "상품A"}
    assert repo.is_product_changed(existing, 10000, "상품B") is True


def test_is_product_changed_both_changed(
    mock_settings: MagicMock, mock_qdrant: AsyncMock
) -> None:
    """가격+제목 모두 변경."""
    repo = CommerceRepository(qdrant_client=mock_qdrant)
    existing = {"price": 10000, "title": "상품A"}
    assert repo.is_product_changed(existing, 20000, "상품B") is True


# ── upsert 테스트 ──


@pytest.mark.asyncio
async def test_upsert_success(mock_settings: MagicMock, mock_qdrant: AsyncMock) -> None:
    """Qdrant upsert 성공 → True."""
    mock_qdrant.upsert = AsyncMock(return_value=None)

    repo = CommerceRepository(qdrant_client=mock_qdrant)
    result = await repo.upsert("12345", [0.1] * 4096, {"title": "test"})

    assert result is True
    mock_qdrant.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_upsert_failure(mock_settings: MagicMock, mock_qdrant: AsyncMock) -> None:
    """Qdrant upsert 실패 → False."""
    mock_qdrant.upsert = AsyncMock(side_effect=Exception("upsert failed"))

    repo = CommerceRepository(qdrant_client=mock_qdrant)
    result = await repo.upsert("12345", [0.1] * 4096, {"title": "test"})

    assert result is False


# ── update_payload 테스트 ──


@pytest.mark.asyncio
async def test_update_payload_success(
    mock_settings: MagicMock, mock_qdrant: AsyncMock
) -> None:
    """payload 업데이트 성공."""
    mock_qdrant.set_payload = AsyncMock(return_value=None)

    repo = CommerceRepository(qdrant_client=mock_qdrant)
    await repo.update_payload("12345", {"price": 20000})

    mock_qdrant.set_payload.assert_called_once()
