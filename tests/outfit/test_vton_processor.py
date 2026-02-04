from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.outfit.schemas import Outfit, OutfitItem, OutfitResponse, UploadSlot
from app.outfit.vton_client import VTONResponse
from app.outfit.vton_processor import VTONProcessor


@pytest.fixture
def mock_vton_client() -> MagicMock:
    client = MagicMock()
    client.generate_outfit_image = AsyncMock(
        return_value=VTONResponse(status="completed", image_data=b"generated-image")
    )
    return client


@pytest.fixture
def processor(mock_vton_client: MagicMock) -> VTONProcessor:
    return VTONProcessor(vton_client=mock_vton_client)


@pytest.fixture
def sample_outfit_response() -> OutfitResponse:
    return OutfitResponse(
        query_summary="테스트 코디",
        outfits=[
            Outfit(
                outfit_id="outfit-1",
                description="테스트",
                clothes_ids=[1, 2],
                items=[
                    OutfitItem(
                        clothes_id=1,
                        image_url="http://img.com/1.jpg",
                        category="TOP",
                        role="상의",
                    ),
                    OutfitItem(
                        clothes_id=2,
                        image_url="http://img.com/2.jpg",
                        category="BOTTOM",
                        role="하의",
                    ),
                ],
            )
        ],
    )


class TestVTONProcessor:
    @pytest.mark.asyncio
    @patch("app.outfit.vton_processor.upload_to_s3")
    async def test_process_success(
        self,
        mock_upload: AsyncMock,
        processor: VTONProcessor,
        mock_vton_client: MagicMock,
        sample_outfit_response: OutfitResponse,
    ) -> None:
        # Arrange
        slot = UploadSlot(file_id=123, object_key="key", presigned_url="http://s3.url")
        upload_slots = [slot]
        mock_upload.return_value = True

        # Act
        await processor.process(sample_outfit_response, upload_slots)

        # Assert
        outfit = sample_outfit_response.outfits[0]
        assert outfit.file_id == 123
        assert outfit.vton_error is None
        mock_vton_client.generate_outfit_image.assert_awaited_once()
        mock_upload.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("app.outfit.vton_processor.upload_to_s3")
    async def test_process_failed_generation(
        self,
        mock_upload: AsyncMock,
        processor: VTONProcessor,
        mock_vton_client: MagicMock,
        sample_outfit_response: OutfitResponse,
    ) -> None:
        # Arrange
        mock_vton_client.generate_outfit_image.return_value = VTONResponse(
            status="failed", error="Generation Error"
        )
        slot = UploadSlot(file_id=123, object_key="key", presigned_url="http://s3.url")
        # Act
        await processor.process(sample_outfit_response, [slot])

        # Assert
        outfit = sample_outfit_response.outfits[0]
        assert outfit.file_id is None
        assert "Generation Error" in outfit.vton_error
        mock_upload.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.outfit.vton_processor.upload_to_s3")
    async def test_process_failed_upload(
        self,
        mock_upload: AsyncMock,
        processor: VTONProcessor,
        sample_outfit_response: OutfitResponse,
    ) -> None:
        # Arrange
        mock_upload.side_effect = Exception("S3 Error")
        slot = UploadSlot(file_id=123, object_key="key", presigned_url="http://s3.url")

        # Act
        await processor.process(sample_outfit_response, [slot])

        # Assert
        outfit = sample_outfit_response.outfits[0]
        assert outfit.file_id is None
        assert "S3 업로드 실패" in outfit.vton_error

    @pytest.mark.asyncio
    async def test_process_not_enough_slots(
        self,
        processor: VTONProcessor,
        sample_outfit_response: OutfitResponse,
        mock_vton_client: MagicMock,
    ) -> None:
        # Arrange
        upload_slots = []  # No slots

        # Act
        await processor.process(sample_outfit_response, upload_slots)

        # Assert
        # Should finish without error, but no VTON done
        mock_vton_client.generate_outfit_image.assert_not_called()
