import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
logger = logging.getLogger("ai-server")

app = FastAPI(title="FFS AI Server (GCP)", version="1.0.0")

# ============================================================
# 글로벌 모델 인스턴스
# ============================================================
validator: ImageValidator = None
nsfw_validator: NSFWValidator = None
fashion_classifier: FashionClassifier = None
fashion_embedder: FashionEmbedder = None


@app.on_event("startup")
def load_models() -> None:
    """서버 시작 시 모델 로드"""
    global validator, nsfw_validator, fashion_classifier, fashion_embedder

    logger.info("모델 초기화 시작...")
    validator = ImageValidator(lazy_load=False)
    nsfw_validator = validator.nsfw_validator
    fashion_classifier = validator.fashion_classifier
    fashion_embedder = validator.fashion_embedder
    logger.info("모델 초기화 완료!")


# ============================================================
# API 스키마
# ============================================================
class ValidateRequest(BaseModel):
    action: str = "validate"  # validate, nsfw, fashion, embedding
    images: list[str]
    user_id: int | None = None


class ValidationResult(BaseModel):
    url: str
    nsfw: dict | None = None
    fashion: dict | None = None
    embedding: list[float] | None = None
    error: str | None = None


class ValidateResponse(BaseModel):
    results: list[ValidationResult]


# ============================================================
# API 엔드포인트
# ============================================================
@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "model_loaded": validator is not None}


@app.post("/validate", response_model=ValidateResponse)
async def process_images(request: ValidateRequest) -> ValidateResponse:
    """이미지 처리 (검증, 분류, 임베딩)"""
    logger.info(
        f"요청 수신 - action: {request.action}, images: {len(request.images)}개"
    )

    if not request.images:
        raise HTTPException(status_code=400, detail="이미지 URL이 없습니다.")

    results = []

    for image_url in request.images:
        try:
            res = _process_single_image(image_url, request.action)
            results.append(res)
        except Exception as e:
            logger.error(f"이미지 처리 실패: {image_url}, 에러: {e}")
            results.append({"url": image_url, "error": str(e)})

    return {"results": results}


def _process_single_image(image_url: str, action: str) -> dict:
    """단일 이미지 처리 로직 (RunPod handler와 동일)"""
    result = {"url": image_url}

    # 이미지 다운로드
    image = download_image(image_url)
    if image is None:
        result["error"] = "IMAGE_DOWNLOAD_FAILED"
        return result

    if action == "validate":
        # 전체 검증
        nsfw_res = nsfw_validator.check(image)
        result["nsfw"] = {
            "is_nsfw": nsfw_res["is_nsfw"],
            "score": nsfw_res["nsfw_score"],
        }

        if not nsfw_res["is_nsfw"]:
            fashion_res = fashion_classifier.check(image)
            result["fashion"] = {
                "is_fashion": fashion_res["is_fashion"],
                "score": fashion_res["fashion_score"],
            }

            if fashion_res["is_fashion"]:
                embedding = fashion_embedder.get_embedding(image)
                result["embedding"] = embedding

    elif action == "nsfw":
        nsfw_res = nsfw_validator.check(image)
        result["nsfw"] = {
            "is_nsfw": nsfw_res["is_nsfw"],
            "score": nsfw_res["nsfw_score"],
        }

    elif action == "fashion":
        fashion_res = fashion_classifier.check(image)
        result["fashion"] = {
            "is_fashion": fashion_res["is_fashion"],
            "score": fashion_res["fashion_score"],
        }

    elif action == "embedding":
        embedding = fashion_embedder.get_embedding(image)
        result["embedding"] = embedding

    return result


if __name__ == "__main__":
    # 로컬 디버깅용
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
