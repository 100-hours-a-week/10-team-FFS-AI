from __future__ import annotations

import io
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from app.closet.background_removal import BackgroundRemover, get_background_remover


class TestBackgroundRemover:
    @pytest.fixture
    def mock_settings(self) -> Generator[MagicMock, None, None]:
        with patch("app.closet.background_removal.settings") as mock_settings:
            mock_settings.ai_model_server_url = "http://test-server:8000"
            yield mock_settings

    @pytest.fixture
    def remover(self, mock_settings: MagicMock) -> BackgroundRemover:
        # Re-initialize to pick up mocked settings
        return BackgroundRemover()

    @pytest.mark.asyncio
    async def test_remove_background_success(self, remover: BackgroundRemover) -> None:
        """배경 제거 성공 케이스 (HTTP Mock)"""
        # Given
        dummy_image = Image.new("RGB", (100, 100), color="white")

        # Prepare Mock Response (Return a valid PNG image)
        mock_output = Image.new("RGBA", (100, 100), color=(255, 255, 255, 0))
        output_buf = io.BytesIO()
        mock_output.save(output_buf, format="PNG")
        output_bytes = output_buf.getvalue()

        # Mock httpx.AsyncClient
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = output_bytes
            mock_client.post = AsyncMock(return_value=mock_response)

            # When
            result_image = await remover.remove_background(dummy_image)

            # Then
            # 1. URL Check
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert str(call_args[0][0]) == "http://test-server:8000/segmentation"

            # 2. Result Check
            assert isinstance(result_image, Image.Image)
            assert result_image.mode == "RGBA"
            assert result_image.size == (100, 100)

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_remove_background_error(self, remover: BackgroundRemover) -> None:
        """배경 제거 실패 시 원본 반환 (Fail Open)"""
        # Given
        dummy_image = Image.new("RGB", (50, 50), color="red")

        # Mock httpx Error
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.__aenter__.return_value = mock_client

            # Simulate Exception
            mock_client.post = AsyncMock(side_effect=Exception("Connection Error"))

            # When
            result_image = await remover.remove_background(dummy_image)

            # Then
            # Should return original image converted to RGBA
            assert result_image.mode == "RGBA"
            assert result_image.size == (50, 50)
            # Should look like original (red)
            assert result_image.getpixel((0, 0)) == (255, 0, 0, 255)

    def test_singleton_pattern(self) -> None:
        """get_background_remover가 싱글톤을 반환하는지 확인"""
        remover1 = get_background_remover()
        remover2 = get_background_remover()
        assert remover1 is remover2
