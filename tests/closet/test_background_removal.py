from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.closet.background_removal import BackgroundRemover, get_background_remover


class TestBackgroundRemover:
    @pytest.fixture
    def mock_seg_model(self: TestBackgroundRemover) -> Generator[MagicMock, None, None]:
        with patch("app.core.models.SegmentationModel") as mock_model_cls:
            model_instance = mock_model_cls.return_value
            model_instance.device = "cpu"
            model_instance._loaded = False
            yield model_instance

    @pytest.fixture
    def remover(
        self: TestBackgroundRemover, mock_seg_model: MagicMock
    ) -> BackgroundRemover:
        # Reset singleton if needed, though we test class directly mostly
        return BackgroundRemover()

    def test_initialization(
        self: TestBackgroundRemover,
        remover: BackgroundRemover,
        mock_seg_model: MagicMock,
    ) -> None:
        """초기화 시 SegmentationModel이 생성되는지 확인"""
        assert remover.seg_model == mock_seg_model

    def test_remove_background_flow(
        self: TestBackgroundRemover,
        remover: BackgroundRemover,
        mock_seg_model: MagicMock,
    ) -> None:
        """배경 제거 전체 로직(전처리->추론->후처리) 테스트"""
        # Given
        dummy_image = Image.new("RGB", (100, 100), color="white")

        # Mocking Prediction Result (Sigmoid applied tensor)
        # Output shape: [1, 1024, 1024] assumption based on logic
        import torch

        mock_pred = torch.rand((1, 1024, 1024))
        mock_seg_model.predict.return_value = mock_pred

        # When
        result_image = remover.remove_background(dummy_image)

        # Then
        # 1. Check if model was loaded
        mock_seg_model.load_model.assert_called_once()

        # 2. Check prediction call
        mock_seg_model.predict.assert_called_once()

        # 3. Check result type and properties
        assert isinstance(result_image, Image.Image)
        assert result_image.mode == "RGBA"  # Alpha channel added
        assert result_image.size == (100, 100)  # Resized back to original

    def test_singleton_pattern(self: TestBackgroundRemover) -> None:
        """get_background_remover가 싱글톤을 반환하는지 확인"""
        remover1 = get_background_remover()
        remover2 = get_background_remover()
        assert remover1 is remover2
