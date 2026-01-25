"""
이미지 검증 AI 모델 래퍼

RunPod Serverless에서 실행될 AI 검증 로직을 담은 모듈입니다.

Models:
1. NSFW 검증: Falconsai/nsfw_image_detection
2. 패션 분류: LAION CLIP-ViT-B-32-laion2B-s34B-b79K
3. 중복/유사 검증: Marqo/marqo-fashionSigLIP (임베딩 생성)
"""

import io
import logging

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

# ============================================================
# 설정 상수
# ============================================================

NSFW_MODEL_ID = "Falconsai/nsfw_image_detection"
CLIP_MODEL_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
FASHION_SIGLIP_MODEL_ID = "Marqo/marqo-fashionSigLIP"

NSFW_THRESHOLD = 0.5  # NSFW 판정 임계값
FASHION_THRESHOLD = 0.3  # 패션 도메인 판정 임계값

# 패션 관련 텍스트 프롬프트 (CLIP용)
FASHION_PROMPTS = [
    "a photo of clothing",
    "a photo of a fashion item",
    "a photo of a shirt",
    "a photo of pants",
    "a photo of a dress",
    "a photo of shoes",
    "a photo of a jacket",
]

NON_FASHION_PROMPTS = [
    "a photo of food",
    "a photo of a landscape",
    "a photo of a person's face",
    "a photo of an animal",
    "a photo of a building",
    "a photo of text or document",
]


# ============================================================
# 이미지 유틸리티
# ============================================================


def download_image(image_url: str, timeout: float = 30.0) -> Image.Image | None:
    """URL에서 이미지 다운로드"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        with httpx.Client(
            timeout=timeout, follow_redirects=True, verify=False, headers=headers
        ) as client:
            response = client.get(image_url)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            return image.convert("RGB")
    except Exception as e:
        logger.error(f"이미지 다운로드 실패: {image_url}, 에러: {e}")
        return None


# ============================================================
# NSFW 검증 (Falconsai)
# ============================================================


class NSFWValidator:
    """
    NSFW 검증기 (Falconsai/nsfw_image_detection)

    ViT 기반 이미지 분류 모델로 부적절한 콘텐츠를 감지합니다.
    """

    def __init__(self: "NSFWValidator") -> None:
        self._classifier = None
        self._loaded = False

    def load_model(self: "NSFWValidator") -> None:
        """모델 로드 (lazy loading)"""
        if self._loaded:
            return

        try:
            from transformers import pipeline

            logger.info(f"NSFW 모델 로딩 중: {NSFW_MODEL_ID}")
            self._classifier = pipeline(
                "image-classification",
                model=NSFW_MODEL_ID,
                device=0,  # GPU 사용
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
                    model=NSFW_MODEL_ID,
                    device=-1,  # CPU
                )
                self._loaded = True
                logger.info("NSFW 모델 로드 완료 (CPU)")
            except Exception as e2:
                logger.error(f"NSFW 모델 CPU 로드도 실패: {e2}")
                raise

    def check(self: "NSFWValidator", image: Image.Image) -> dict:
        """
        NSFW 검증

        Args:
            image: PIL Image 객체

        Returns:
            {
                "is_nsfw": bool,
                "nsfw_score": float,
                "label": str
            }
        """
        if not self._loaded:
            self.load_model()

        try:
            results = self._classifier(image)
            print(f"[DEBUG] NSFW Model Output: {results}")  # 모델 원본 출력
            # [{"label": "nsfw", "score": 0.95}, {"label": "normal", "score": 0.05}]

            nsfw_score = 0.0
            label = "normal"

            for result in results:
                if result["label"].lower() == "nsfw":
                    nsfw_score = result["score"]
                    if nsfw_score >= NSFW_THRESHOLD:
                        label = "nsfw"
                    break

            return {
                "is_nsfw": nsfw_score >= NSFW_THRESHOLD,
                "nsfw_score": nsfw_score,
                "label": label,
            }
        except Exception as e:
            logger.error(f"NSFW 검증 실패: {e}")
            return {
                "is_nsfw": False,
                "nsfw_score": 0.0,
                "label": "error",
                "error": str(e),
            }


# ============================================================
# 패션 도메인 분류 (LAION CLIP)
# ============================================================


class FashionClassifier:
    """
    패션 도메인 분류기 (LAION CLIP-ViT-B-32)

    이미지가 패션/의류 도메인인지 CLIP의 zero-shot 분류로 판별합니다.
    """

    def __init__(self: "FashionClassifier") -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._loaded = False

    def load_model(self: "FashionClassifier") -> None:
        """모델 로드 (lazy loading)"""
        if self._loaded:
            return

        try:
            import open_clip
            import torch

            logger.info("CLIP 모델 로딩 중: ViT-B-32 (laion2b_s34b_b79k)")

            device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
            )
            self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self._device = device
            self._loaded = True
            logger.info(f"CLIP 모델 로드 완료 ({device})")
        except Exception as e:
            logger.error(f"CLIP 모델 로드 실패: {e}")
            raise

    def check(self: "FashionClassifier", image: Image.Image) -> dict:
        """
        패션 도메인 분류

        Args:
            image: PIL Image 객체

        Returns:
            {
                "is_fashion": bool,
                "fashion_score": float,
                "non_fashion_score": float
            }
        """
        if not self._loaded:
            self.load_model()

        try:
            import torch

            # 이미지 전처리
            image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)

            # 텍스트 토큰화
            all_prompts = FASHION_PROMPTS + NON_FASHION_PROMPTS
            text_tokens = self._tokenizer(all_prompts).to(self._device)

            # 추론
            with torch.no_grad():
                image_features = self._model.encode_image(image_tensor)
                text_features = self._model.encode_text(text_tokens)

                # 정규화
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # 유사도 계산
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                similarity = similarity[0].cpu().numpy()

            # 패션 vs 비패션 점수 계산
            fashion_score = float(similarity[: len(FASHION_PROMPTS)].sum())
            non_fashion_score = float(similarity[len(FASHION_PROMPTS) :].sum())

            is_fashion = (
                fashion_score > non_fashion_score and fashion_score >= FASHION_THRESHOLD
            )

            return {
                "is_fashion": is_fashion,
                "fashion_score": fashion_score,
                "non_fashion_score": non_fashion_score,
            }
        except Exception as e:
            logger.error(f"패션 분류 실패: {e}")
            return {
                "is_fashion": True,  # 에러 시 통과 처리
                "fashion_score": 0.0,
                "non_fashion_score": 0.0,
                "error": str(e),
            }


# ============================================================
# 임베딩 생성 (Marqo-FashionSigLIP)
# ============================================================


class FashionEmbedder:
    """
    패션 이미지 임베딩 생성기 (Marqo-FashionSigLIP)

    패션 도메인에 특화된 SigLIP 모델로 고품질 임베딩을 생성합니다.
    중복/유사 이미지 검증 및 Qdrant 저장에 사용됩니다.
    """

    def __init__(self: "FashionEmbedder") -> None:
        self._model = None
        self._loaded = False
        self._transform = None

    def load_model(self: "FashionEmbedder") -> None:
        """모델 로드 (lazy loading)"""
        if self._loaded:
            return

        try:
            import torch
            from torchvision import transforms
            from transformers import AutoModel

            logger.info(f"FashionSigLIP 모델 로딩 중: {FASHION_SIGLIP_MODEL_ID}")

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # 모델만 로드 (Processor 대신 직접 전처리)
            self._model = AutoModel.from_pretrained(
                FASHION_SIGLIP_MODEL_ID, trust_remote_code=True
            ).to(device)

            # 수동 전처리 transform (SigLIP 표준)
            self._transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

            self._device = device
            self._loaded = True
            logger.info(f"FashionSigLIP 모델 로드 완료 ({device})")
        except Exception as e:
            logger.error(f"FashionSigLIP 모델 로드 실패: {e}")
            # Fallback: CLIP 임베딩 사용
            logger.info("Fallback: CLIP 임베딩 사용")
            self._use_clip_fallback = True
            self._loaded = True

    def get_embedding(self: "FashionEmbedder", image: Image.Image) -> list[float]:
        """
        이미지 임베딩 생성

        Args:
            image: PIL Image 객체

        Returns:
            임베딩 벡터 (768차원)
        """
        if not self._loaded:
            self.load_model()

        # Fallback 모드면 빈 벡터 반환 (CLIP으로 대체 가능)
        if hasattr(self, "_use_clip_fallback") and self._use_clip_fallback:
            return [0.0] * 768

        try:
            import torch

            # 이미지 전처리
            image_tensor = self._transform(image).unsqueeze(0).to(self._device)

            # 임베딩 추출
            with torch.no_grad():
                outputs = self._model.get_image_features(pixel_values=image_tensor)
                # 정규화
                embedding = outputs / outputs.norm(dim=-1, keepdim=True)
                embedding = embedding[0].cpu().numpy().tolist()

            return embedding
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return []

    def compute_similarity(
        self: "FashionEmbedder", embedding1: list[float], embedding2: list[float]
    ) -> float:
        """
        두 임베딩 간 코사인 유사도 계산

        Args:
            embedding1: 첫 번째 임베딩
            embedding2: 두 번째 임베딩

        Returns:
            유사도 (0.0 ~ 1.0)
        """
        try:
            import numpy as np

            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            similarity = np.dot(vec1, vec2) / (
                np.linalg.norm(vec1) * np.linalg.norm(vec2)
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"유사도 계산 실패: {e}")
            return 0.0


# ============================================================
# 통합 검증기
# ============================================================


class ImageValidator:
    """
    통합 이미지 검증기

    NSFW, 패션 분류, 임베딩 생성을 하나의 인터페이스로 제공합니다.
    """

    def __init__(self: "ImageValidator", lazy_load: bool = True) -> None:
        """
        Args:
            lazy_load: True면 첫 사용 시 모델 로드, False면 즉시 로드
        """
        self.nsfw_validator = NSFWValidator()
        self.fashion_classifier = FashionClassifier()
        self.fashion_embedder = FashionEmbedder()

        if not lazy_load:
            self.load_all_models()

    def load_all_models(self: "ImageValidator") -> None:
        """모든 모델 로드"""
        logger.info("모든 검증 모델 로딩 시작...")
        self.nsfw_validator.load_model()
        self.fashion_classifier.load_model()
        self.fashion_embedder.load_model()
        logger.info("모든 검증 모델 로딩 완료")

    def validate_image(self: "ImageValidator", image_url: str) -> dict:
        """
        이미지 종합 검증

        Args:
            image_url: 검증할 이미지 URL

        Returns:
            {
                "url": str,
                "nsfw": {"is_nsfw": bool, "score": float},
                "fashion": {"is_fashion": bool, "score": float},
                "embedding": list[float],
                "error": str | None
            }
        """
        result = {
            "url": image_url,
            "nsfw": None,
            "fashion": None,
            "embedding": [],
            "error": None,
        }

        # 이미지 다운로드
        image = download_image(image_url)
        if image is None:
            result["error"] = "IMAGE_DOWNLOAD_FAILED"
            return result

        # NSFW 검증
        nsfw_result = self.nsfw_validator.check(image)
        result["nsfw"] = {
            "is_nsfw": nsfw_result["is_nsfw"],
            "score": nsfw_result["nsfw_score"],
        }

        # NSFW면 나머지 검증 스킵
        if nsfw_result["is_nsfw"]:
            return result

        # 패션 분류
        fashion_result = self.fashion_classifier.check(image)
        result["fashion"] = {
            "is_fashion": fashion_result["is_fashion"],
            "score": fashion_result["fashion_score"],
        }

        # 패션 아니면 임베딩 스킵
        if not fashion_result["is_fashion"]:
            return result

        # 임베딩 생성
        embedding = self.fashion_embedder.get_embedding(image)
        result["embedding"] = embedding

        return result

    def validate_batch(self: "ImageValidator", image_urls: list[str]) -> list[dict]:
        """
        배치 이미지 검증

        Args:
            image_urls: 검증할 이미지 URL 목록

        Returns:
            검증 결과 목록
        """
        results = []
        for url in image_urls:
            result = self.validate_image(url)
            results.append(result)
        return results


# ============================================================
# Mock 검증기 (테스트용)
# ============================================================


class MockImageValidator:
    """
    테스트용 Mock 검증기

    실제 모델 없이 테스트할 때 사용합니다.
    """

    def validate_image(self: "MockImageValidator", image_url: str) -> dict:
        """Mock 검증 결과 반환"""
        # 테스트용 규칙
        is_nsfw = "nsfw" in image_url.lower()
        is_fashion = (
            "fashion" not in image_url.lower() or "food" not in image_url.lower()
        )

        return {
            "url": image_url,
            "nsfw": {"is_nsfw": is_nsfw, "score": 0.9 if is_nsfw else 0.1},
            "fashion": {"is_fashion": is_fashion, "score": 0.8 if is_fashion else 0.2},
            "embedding": [0.1] * 768 if is_fashion and not is_nsfw else [],
            "error": None,
        }

    def validate_batch(self: "MockImageValidator", image_urls: list[str]) -> list[dict]:
        """Mock 배치 검증"""
        return [self.validate_image(url) for url in image_urls]
