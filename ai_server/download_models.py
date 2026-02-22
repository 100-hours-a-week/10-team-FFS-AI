"""Korean-Bllossom-Vision-8B 사전 다운로드 스크립트

GCP 인스턴스에서 vLLM 서버 시작 전에 실행하여
HuggingFace 모델을 미리 캐싱합니다.

사용법:
    python download_models.py
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Bllossom/llama-3.1-Korean-Bllossom-Vision-8B"


def download() -> None:
    logger.info(f"Downloading model: {MODEL_NAME} ...")

    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME)
    logger.info("Model download complete.")


if __name__ == "__main__":
    download()
