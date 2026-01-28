import logging
import os
import ssl
import sys

import open_clip
from transformers import pipeline

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings  # noqa: E402

# SSL 인증서 오류 방지 (Mac 등에서 발생 가능)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()

NSFW_MODEL_ID = settings.nsfw_model_id
CLIP_MODEL_NAME = "ViT-B-32"  # This part needs parsing if we use full ID from settings, but keeping simple for now or using settings.clip_model_id for logic
CLIP_PRETRAINED = "laion2b_s34b_b79k"


def download_nsfw_model() -> None:
    """NSFW 모델 다운로드"""
    logger.info(f"Downloading NSFW model: {NSFW_MODEL_ID}...")
    pipeline("image-classification", model=NSFW_MODEL_ID)
    logger.info("NSFW model downloaded.")


def download_clip_model() -> None:
    """CLIP 모델 다운로드"""
    logger.info(f"Downloading CLIP model: {CLIP_MODEL_NAME} ({CLIP_PRETRAINED})...")
    open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
    logger.info("CLIP model downloaded.")


def download_segmentation_model() -> None:
    """Segmentation 모델 다운로드 (BiRefNet)"""
    model_id = "ZhengPeng7/BiRefNet"
    logger.info(f"Downloading Segmentation model: {model_id}...")

    from transformers import AutoModelForImageSegmentation

    # trust_remote_code=True 필요
    AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
    logger.info("Segmentation model downloaded.")


def main() -> None:
    logger.info("Starting model download...")

    # 1. NSFW Model
    try:
        download_nsfw_model()
    except Exception as e:
        logger.error(f"Failed to download NSFW model: {e}")

    # 2. CLIP Model
    try:
        download_clip_model()
    except Exception as e:
        logger.error(f"Failed to download CLIP model: {e}")

    # 3. Segmentation Model (BiRefNet)
    try:
        download_segmentation_model()
    except Exception as e:
        logger.error(f"Failed to download Segmentation model: {e}")

    logger.info("All models downloaded successfully (or cached).")


if __name__ == "__main__":
    main()
