"""
RunPod Serverless Handler

이미지 검증을 위한 RunPod Serverless 진입점입니다.

Supported Actions:
- "validate": 이미지 검증 (NSFW, 패션 분류, 임베딩)
- "nsfw": NSFW 검증만
- "fashion": 패션 분류만
- "embedding": 임베딩 생성만
"""

import logging

from validators import (
    FashionClassifier,
    FashionEmbedder,
    ImageValidator,
    NSFWValidator,
    download_image,
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# RunPod import (optional)
try:
    import runpod
except ImportError:
    logger.warning(
        "runpod 패키지가 설치되지 않았습니다. 로컬 테스트 모드로 실행됩니다."
    )
    runpod = None

# ============================================================
# 글로벌 모델 인스턴스 (Cold Start 최소화)
# ============================================================

validator: ImageValidator = None
nsfw_validator: NSFWValidator = None
fashion_classifier: FashionClassifier = None
fashion_embedder: FashionEmbedder = None


def load_models():
    """모델 초기화 (서버 시작 시 1회)"""
    global validator, nsfw_validator, fashion_classifier, fashion_embedder

    logger.info("모델 초기화 시작...")

    validator = ImageValidator(lazy_load=False)
    nsfw_validator = validator.nsfw_validator
    fashion_classifier = validator.fashion_classifier
    fashion_embedder = validator.fashion_embedder

    logger.info("모델 초기화 완료!")


# ============================================================
# 핸들러 함수
# ============================================================


def handler(job: dict) -> dict:
    """
    RunPod Serverless 핸들러

    Input:
        {
            "input": {
                "action": "validate" | "nsfw" | "fashion" | "embedding",
                "images": ["url1", "url2", ...],
                "user_id": 123  # (optional) 중복 검증용
            }
        }

    Output:
        {
            "results": [
                {
                    "url": "url1",
                    "nsfw": {"is_nsfw": false, "score": 0.1},
                    "fashion": {"is_fashion": true, "score": 0.8},
                    "embedding": [0.1, 0.2, ...]
                },
                ...
            ]
        }
    """
    global validator

    try:
        job_input = job.get("input", {})
        action = job_input.get("action", "validate")
        images = job_input.get("images", [])
        user_id = job_input.get("user_id")

        logger.info(
            f"작업 시작 - action: {action}, images: {len(images)}개, user_id: {user_id}"
        )

        if not images:
            return {"error": "이미지 URL이 없습니다", "results": []}

        # 모델이 로드되지 않았으면 로드
        if validator is None:
            load_models()

        results = []

        for image_url in images:
            try:
                result = process_single_image(image_url, action)
                results.append(result)
            except Exception as e:
                logger.error(f"이미지 처리 실패: {image_url}, 에러: {e}")
                results.append({"url": image_url, "error": str(e)})

        logger.info(f"작업 완료 - 처리된 이미지: {len(results)}개")

        return {"results": results}

    except Exception as e:
        logger.error(f"핸들러 에러: {e}")
        return {"error": str(e), "results": []}


def process_single_image(image_url: str, action: str) -> dict:
    """
    단일 이미지 처리

    Args:
        image_url: 이미지 URL
        action: 처리 액션

    Returns:
        처리 결과
    """
    result = {"url": image_url}

    # 이미지 다운로드
    image = download_image(image_url)
    if image is None:
        result["error"] = "IMAGE_DOWNLOAD_FAILED"
        return result

    if action == "validate":
        # 전체 검증
        nsfw_result = nsfw_validator.check(image)
        result["nsfw"] = {
            "is_nsfw": nsfw_result["is_nsfw"],
            "score": nsfw_result["nsfw_score"],
        }

        if not nsfw_result["is_nsfw"]:
            fashion_result = fashion_classifier.check(image)
            result["fashion"] = {
                "is_fashion": fashion_result["is_fashion"],
                "score": fashion_result["fashion_score"],
            }

            if fashion_result["is_fashion"]:
                embedding = fashion_embedder.get_embedding(image)
                result["embedding"] = embedding

    elif action == "nsfw":
        # NSFW 검증만
        nsfw_result = nsfw_validator.check(image)
        result["nsfw"] = {
            "is_nsfw": nsfw_result["is_nsfw"],
            "score": nsfw_result["nsfw_score"],
        }

    elif action == "fashion":
        # 패션 분류만
        fashion_result = fashion_classifier.check(image)
        result["fashion"] = {
            "is_fashion": fashion_result["is_fashion"],
            "score": fashion_result["fashion_score"],
        }

    elif action == "embedding":
        # 임베딩 생성만
        embedding = fashion_embedder.get_embedding(image)
        result["embedding"] = embedding

    else:
        result["error"] = f"알 수 없는 action: {action}"

    return result


# ============================================================
# RunPod 서버 시작
# ============================================================

if __name__ == "__main__":
    logger.info("RunPod Handler 시작...")

    # 모델 사전 로드 (Cold Start 방지)
    load_models()

    if runpod:
        runpod.serverless.start({"handler": handler})
    else:
        # 로컬 테스트
        logger.info("로컬 테스트 모드")
        test_job = {
            "input": {"action": "validate", "images": ["https://example.com/test.jpg"]}
        }
        result = handler(test_job)
        print(result)
