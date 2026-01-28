"""
배경 제거 (Background Removal) 모듈

BiRefNet(Bilateral Reference Network)을 사용하여 고품질 배경 제거를 수행합니다.
Business Logic Layer: 이미지 전처리, 후처리 담당
Model Infrastructure Layer: app.core.models.SegmentationModel (모델 로딩 및 추론)
"""

from __future__ import annotations

import logging

from PIL import Image
from torchvision import transforms

from app.core.models import SegmentationModel

logger = logging.getLogger(__name__)


class BackgroundRemover:
    """
    BiRefNet 기반 배경 제거 클래스 (Business Logic)
    """

    def __init__(self: BackgroundRemover) -> None:
        self.seg_model = SegmentationModel()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def remove_background(self: BackgroundRemover, image: Image.Image) -> Image.Image:
        """
        이미지 배경 제거

        Args:
            image: PIL Image 객체 (RGB)

        Returns:
            배경이 제거된 PIL Image 객체 (RGBA)
        """
        # 모델 로드 (Lazy Loading은 SegmentationModel 내부에서 처리)
        # 단, device 정보를 알기 위해 load_model() 호출 필요할 수 있음
        # predict() 호출 시 내부적으로 load하므로 바로 predict 호출

        original_size = image.size

        # 전처리
        # 모델이 로드되어야 device를 알 수 있으므로, predict 호출 직전에 로드 확인
        # (SegmentationModel.device 프로퍼티 접근 접근 시 로드 필요하면 로드)
        if not self.seg_model._loaded:
            self.seg_model.load_model()

        device = self.seg_model.device
        input_images = self.transform_image(image).unsqueeze(0).to(device)

        # 추론 (Infrastructure Layer 위임)
        # preds: Tensor (Sigmoid 적용됨)
        preds = self.seg_model.predict(input_images)

        # 마스크 후처리
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred).resize(original_size)

        # 이미지 합성 (Alpha Channel 적용)
        image.putalpha(pred_pil)

        return image


# 싱글톤 인스턴스
_remover_instance: BackgroundRemover | None = None


def get_background_remover() -> BackgroundRemover:
    global _remover_instance
    if _remover_instance is None:
        _remover_instance = BackgroundRemover()
    return _remover_instance
