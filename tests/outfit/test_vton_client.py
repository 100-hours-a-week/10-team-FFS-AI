from unittest.mock import MagicMock, patch

import pytest

from app.outfit.vton_client import VTONClient, VTONRequest


@pytest.fixture
def vton_client() -> VTONClient:
    with patch("app.config.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            gemini_api_key="test-key",
            vton_model="gemini-3-pro-image-preview",
            vton_fallback_model="gemini-2.5-flash-image",
        )
        return VTONClient()


class TestVTONClient:
    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.get")
    async def test_download_image_success(
        self, mock_get: MagicMock, vton_client: VTONClient
    ) -> None:
        mock_get.return_value = MagicMock(status_code=200, content=b"fake-image-data")

        result = await vton_client._download_image("https://example.com/image.jpg")

        assert result == b"fake-image-data"
        mock_get.assert_called_once()

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.get")
    async def test_download_image_failure(
        self, mock_get: MagicMock, vton_client: VTONClient
    ) -> None:
        mock_get.side_effect = Exception("Download error")

        result = await vton_client._download_image("https://example.com/image.jpg")

        assert result is None

    @pytest.mark.asyncio
    @patch.object(VTONClient, "_download_image")
    async def test_load_images(
        self, mock_download: MagicMock, vton_client: VTONClient
    ) -> None:
        mock_download.return_value = b"image-data"

        urls = ["https://ex1.com", "https://ex2.com"]
        parts = await vton_client._load_images(urls)

        assert len(parts) == 2
        assert "inline_data" in parts[0]
        assert parts[0]["inline_data"]["mime_type"] == "image/jpeg"

    @pytest.mark.asyncio
    @patch.object(VTONClient, "_load_images")
    @patch("httpx.AsyncClient.post")
    async def test_generate_outfit_image_success(
        self, mock_post: MagicMock, mock_load: MagicMock, vton_client: VTONClient
    ) -> None:
        mock_load.return_value = [
            {"inline_data": {"data": "base64", "mime_type": "image/jpeg"}}
        ]

        # Mock Gemini API response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "data": "ZmFrZS1nZW5lcmF0ZWQtaW1hZ2U=",
                                    "mime_type": "image/png",
                                }
                            }
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp

        request = VTONRequest(image_urls=["https://test.com/img.jpg"])
        response = await vton_client.generate_outfit_image(request)

        assert response.status == "completed"
        assert response.image_data == b"fake-generated-image"
        assert response.error is None

    @pytest.mark.asyncio
    @patch.object(VTONClient, "_load_images")
    @patch("httpx.AsyncClient.post")
    async def test_generate_outfit_image_api_error(
        self, mock_post: MagicMock, mock_load: MagicMock, vton_client: VTONClient
    ) -> None:
        mock_load.return_value = [
            {"inline_data": {"data": "base64", "mime_type": "image/jpeg"}}
        ]

        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"
        mock_post.return_value = mock_resp

        request = VTONRequest(image_urls=["https://test.com/img.jpg"])
        response = await vton_client.generate_outfit_image(request)

        assert response.status == "failed"
        assert "API Error (400)" in response.error

    @pytest.mark.asyncio
    @patch.object(VTONClient, "_load_images")
    @patch("httpx.AsyncClient.post")
    async def test_generate_outfit_image_fallback_success(
        self, mock_post: MagicMock, mock_load: MagicMock, vton_client: VTONClient
    ) -> None:
        """메인 모델이 503 에러를 낼 경우 Fallback 모델로 우회하여 성공하는지 테스트"""
        mock_load.return_value = [
            {"inline_data": {"data": "base64", "mime_type": "image/jpeg"}}
        ]

        # 1차 시도 (메인 모델): 503 에러 실패 응답
        mock_resp_fail = MagicMock()
        mock_resp_fail.status_code = 503
        mock_resp_fail.text = "Service Unavailable"

        # 2차 시도 (Fallback 모델): 200 성공 응답
        mock_resp_success = MagicMock()
        mock_resp_success.status_code = 200
        mock_resp_success.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "data": "ZmFrZS1nZW5lcmF0ZWQtaW1hZ2U=",
                                    "mime_type": "image/png",
                                }
                            }
                        ]
                    }
                }
            ]
        }

        # httpx post 호출 순서대로 모의 응답 지정
        mock_post.side_effect = [mock_resp_fail, mock_resp_success]

        request = VTONRequest(image_urls=["https://test.com/img.jpg"])
        response = await vton_client.generate_outfit_image(request)

        # 결과 검증
        assert response.status == "completed"
        assert response.image_data == b"fake-generated-image"
        # API 호출이 정확히 2번(1순위 실패 -> 2순위 성공) 일어났는지 확인
        assert mock_post.call_count == 2
