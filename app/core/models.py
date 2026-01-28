"""
AI 모델 로더 (Singleton Pattern)

실제 AI 모델(HuggingFace, OpenCLIP 등)을 로드하고 관리하는 모듈입니다.
비즈니스 로직(검증 등)은 ai_validators.py에서 처리하고, 여기서는 순수 모델 로딩 및 추론만 담당합니다.
"""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class NSFWValidator:
    """
    NSFW 검증기 (Falconsai/nsfw_image_detection)
    ViT 기반 이미지 분류 모델로 부적절한 콘텐츠를 감지합니다.
    """

    def __init__(self: NSFWValidator) -> None:
        self._classifier = None
        self._loaded = False

    def load_model(self: NSFWValidator) -> None:
        """모델 로드 (lazy loading)"""
        if self._loaded:
            return

        try:
            from transformers import pipeline

            logger.info(f"NSFW 모델 로딩 중: {settings.nsfw_model_id}")
            self._classifier = pipeline(
                "image-classification",
                model=settings.nsfw_model_id,
                device=0,  # GPU 사용 (가능하면)
            )
            self._loaded = True
            logger.info("NSFW 모델 로드 완료")
        except Exception as e:
            logger.error(f"NSFW 모델 로드 실패: {e}")
            # CPU fallback
            try:
                from transformers import pipeline

                self._classifier = pipeline(
                    "image-classification",
                    model=settings.nsfw_model_id,
                    device=-1,  # CPU
                )
                self._loaded = True
                logger.info("NSFW 모델 로드 완료 (CPU)")
            except Exception as e2:
                logger.error(f"NSFW 모델 CPU 로드도 실패: {e2}")
                raise

    def predict(self: NSFWValidator, image: Image.Image) -> Any:  # noqa: ANN401
        """추론 실행 (Raw Output 반환)"""
        if not self._loaded:
            self.load_model()
        return self._classifier(image)


class FashionClassifier:
    """
    패션 도메인 분류기 (LAION CLIP-ViT-B-32)
    이미지가 패션/의류 도메인인지 CLIP의 zero-shot 분류로 판별합니다.
    """

    def __init__(self: FashionClassifier) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._loaded = False
        self._device = "cpu"

    def load_model(self: FashionClassifier) -> None:
        """모델 로드 (lazy loading)"""
        if self._loaded:
            return

        try:
            import open_clip
            import torch

            model_name = "ViT-B-32"  # 기본값
            pretrained = "laion2b_s34b_b79k"  # 기본값

            # settings.clip_model_id가 복잡한 문자열일 수 있어 파싱 필요하지만
            # 여기서는 편의상 하드코딩된 model/pretrained를 사용하거나,
            # settings에서 분리해서 받는 것이 좋음.
            # (현재 구조상 호환성을 위해 기존 로직 유지)

            logger.info(f"CLIP 모델 로딩 중: {model_name} ({pretrained})")

            device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
            self._tokenizer = open_clip.get_tokenizer(model_name)
            self._device = device
            self._loaded = True
            logger.info(f"CLIP 모델 로드 완료 ({device})")
        except Exception as e:
            logger.error(f"CLIP 모델 로드 실패: {e}")
            raise

    def get_features(
        self: FashionClassifier, image: Image.Image, texts: list[str]
    ) -> Any:  # noqa: ANN401
        """이미지 및 텍스트 특징 추출"""
        if not self._loaded:
            self.load_model()

        import torch

        image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        text_tokens = self._tokenizer(texts).to(self._device)

        with torch.no_grad():
            image_features = self._model.encode_image(image_tensor)
            text_features = self._model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            return similarity[0].cpu().numpy()


class SegmentationModel:
    """
    배경 제거 모델 (ZhengPeng7/BiRefNet)
    """

    def __init__(self: SegmentationModel) -> None:
        self._model = None
        self._loaded = False
        self._device = "cpu"
        self.model_id = "ZhengPeng7/BiRefNet"

    def load_model(self: SegmentationModel) -> None:
        """모델 로드 (lazy loading)"""
        if self._loaded:
            return

        try:
            import torch
            from transformers import AutoModelForImageSegmentation

            logger.info(f"Segmentation 모델 로딩 중: {self.model_id}")
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = AutoModelForImageSegmentation.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self._model.to(self._device)
            self._model.eval()
            self._loaded = True
            logger.info(f"Segmentation 모델 로드 완료 ({self._device})")
        except Exception as e:
            logger.error(f"Segmentation 모델 로드 실패: {e}")
            raise

    def predict(self: SegmentationModel, input_tensor: Any) -> Any:  # noqa: ANN401
        """추론 실행"""
        if not self._loaded:
            self.load_model()

        import torch

        with torch.no_grad():
            # 모델 output이 리스트인 경우 마지막 요소 사용 (BiRefNet 특성)
            preds = self._model(input_tensor)[-1].sigmoid().cpu()
        return preds

    @property
    def device(self: SegmentationModel) -> str:
        return self._device
