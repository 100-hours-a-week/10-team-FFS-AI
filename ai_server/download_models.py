import logging
import ssl

import open_clip
from transformers import pipeline

# SSL 인증서 오류 방지 (Mac 등에서 발생 가능)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NSFW_MODEL_ID = "Falconsai/nsfw_image_detection"
CLIP_MODEL_NAME = "ViT-B-32"
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

    logger.info("All models downloaded successfully (or cached).")


if __name__ == "__main__":
    main()
