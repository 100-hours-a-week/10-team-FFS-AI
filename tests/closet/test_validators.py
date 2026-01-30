"""
Validator 단위 테스트 - AI 모델 동작 검증

네트워크 없이 Mock Validator 로직만 테스트합니다.
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.closet.validators import ImageValidator, MockImageValidator


# 1. Mock Validator Tests (Legacy/Backup)
def test_mock_validator_nsfw_detection() -> None:
    """Mock Validator NSFW 탐지 테스트"""
    validator = MockImageValidator()
    nsfw_url = "https://example.com/nsfw_image.jpg"
    result = validator.validate_image(nsfw_url)
    assert result["url"] == nsfw_url
    assert result["nsfw"]["is_nsfw"] is True


def test_mock_validator_normal_image() -> None:
    """Mock Validator 정상 이미지 테스트"""
    validator = MockImageValidator()
    normal_url = "https://example.com/shirt.jpg"
    result = validator.validate_image(normal_url)
    # The new Mock logic checks "food" or "landscape" for fashion=False
    # Only checks "nsfw" for nsfw=True
    assert result["nsfw"]["is_nsfw"] is False
    assert result["fashion"]["is_fashion"] is True


def test_mock_validator_batch() -> None:
    """Mock Validator 배치 처리 테스트"""
    validator = MockImageValidator()
    urls = ["https://example.com/shirt.jpg", "https://example.com/nsfw_image.jpg"]
    results = validator.validate_batch(urls)
    assert len(results) == 2
    assert results[0]["fashion"]["is_fashion"] is True
    assert results[1]["nsfw"]["is_nsfw"] is True


# 2. Real ImageValidator Tests (HTTP Mock)
class TestImageValidator:
    """실제 ImageValidator의 HTTP 요청 로직 테스트"""

    @pytest.fixture
    def mock_settings(self) -> Generator[MagicMock, None, None]:
        with patch("app.closet.validators.settings") as mock_settings:
            mock_settings.ai_model_server_url = "http://test-server:8000"
            yield mock_settings

    @pytest.fixture
    def validator(self, mock_settings: MagicMock) -> ImageValidator:
        # We need to mock _download_image_sync in init or make sure init is safe
        # init only checks settings, so it is safe.
        return ImageValidator()

    @pytest.mark.asyncio
    async def test_validate_image_success(self, validator: ImageValidator) -> None:
        """정상 이미지 검증 (NSFW False, Fashion True)"""
        # Given
        url = "http://example.com/test.jpg"

        # Mock Download (Prevent actual HTTP/Disk I/O)
        with patch.object(validator, "_download_image_sync") as mock_download:
            # Return dummy image
            from PIL import Image

            mock_download.return_value = Image.new("RGB", (100, 100))

            # Mock HTTP Calls
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = mock_client_cls.return_value
                mock_client.__aenter__.return_value = mock_client

                # Setup responses for 2 sequential calls: /nsfw then /fashion
                # Call 1: NSFW -> [{"label": "normal", "score": 0.9}]
                nsfw_resp = MagicMock()
                nsfw_resp.status_code = 200
                nsfw_resp.json.return_value = [
                    {"label": "normal", "score": 0.9},
                    {"label": "nsfw", "score": 0.1},
                ]

                # Call 2: Fashion -> [0.9] (FASHION SCORE HIGH)
                # Need to match expected vector size logic in validate_image
                # (7 items + 6 items)
                fashion_scores = [0.8] * 7 + [0.1] * 6
                fashion_resp = MagicMock()
                fashion_resp.status_code = 200
                fashion_resp.json.return_value = fashion_scores

                # Use AsyncMock for .post to make it awaitable
                mock_client.post = AsyncMock(side_effect=[nsfw_resp, fashion_resp])

                # When
                result = await validator.validate_image(url)

                # Then
                assert result["nsfw"]["is_nsfw"] is False
                assert result["fashion"]["is_fashion"] is True
                assert mock_client.post.call_count == 2

                # Verify URL targets
                calls = mock_client.post.call_args_list
                assert str(calls[0][0][0]) == "http://test-server:8000/nsfw"
                assert str(calls[1][0][0]) == "http://test-server:8000/fashion"

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_validate_image_cleanup_nsfw(self, validator: ImageValidator) -> None:
        """NSFW 탐지 시 Fashion 호출 안 함"""
        # Given
        url = "http://example.com/nsfw.jpg"

        with patch.object(validator, "_download_image_sync") as mock_download:
            from PIL import Image

            mock_download.return_value = Image.new("RGB", (100, 100))

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = mock_client_cls.return_value
                mock_client.__aenter__.return_value = mock_client

                # Call 1: NSFW -> [{"label": "nsfw", "score": 0.9}]
                nsfw_resp = MagicMock()
                nsfw_resp.status_code = 200
                nsfw_resp.json.return_value = [{"label": "nsfw", "score": 0.9}]

                # AsyncMock
                mock_client.post = AsyncMock(return_value=nsfw_resp)

                # When
                result = await validator.validate_image(url)

                # Then
                assert result["nsfw"]["is_nsfw"] is True
                # Fashion should be None or not computed
                assert result.get("fashion") is None
                # Should stop after first call
                assert mock_client.post.call_count == 1
                assert (
                    str(mock_client.post.call_args[0][0])
                    == "http://test-server:8000/nsfw"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
