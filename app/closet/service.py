"""
Closet 서비스 - 이미지 검증 및 분석 비즈니스 로직 (API 명세 v2 기준)

주요 기능:
1. 이미지 검증 (Validate): 포맷, 크기, 중복/유사(Marqo-FashionSigLIP), 패션 여부(LAION CLIP), NSFW(Falconsai)
2. 이미지 분석 (Analyze): 배경 제거, AI 속성 분석
3. 작업 상태 관리: 비동기 작업 진행 상태 추적

AI 모델 (RunPod Serverless에서 실행):
- Falconsai/nsfw_image_detection: NSFW 검증
- LAION CLIP-ViT-B-32-laion2b-s34b-b79k: 도메인 탐지 (패션 여부)
- Marqo-FashionSigLIP: 중복/유사 이미지 검증 (임베딩 기반)
"""

import logging
import os
import time
from typing import Optional

import httpx

from app.closet.schemas import (
    AnalysisAttributes,
    # Analyze API
    AnalyzeRequest,
    AnalyzeResponse,
    BatchMeta,
    BatchResultItem,
    BatchStatus,
    # Batch Status API
    BatchStatusResponse,
    TaskStatus,
    # Validate API
    ValidateRequest,
    ValidateResponse,
    ValidationErrorCode,
    ValidationResult,
    ValidationSummary,
)
from app.core.qdrant import QdrantService

logger = logging.getLogger(__name__)

# ============================================================
# 설정 상수
# ============================================================

ALLOWED_FORMATS = {"jpg", "jpeg", "png"}  # webp 제외
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MIN_IMAGE_RESOLUTION = 512  # 최소 512x512
SIMILARITY_THRESHOLD = 0.95  # 유사도 임계값

# RunPod 설정 (환경변수에서 로드)
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

# Mock 모드 (RunPod 없이 테스트용)
USE_MOCK_VALIDATOR = os.getenv("USE_MOCK_VALIDATOR", "true").lower() == "true"


# ============================================================
# 작업 저장소 (실제로는 Redis 등 사용)
# ============================================================

# 임시 인메모리 저장소 (나중에 Redis로 교체)
_batch_store: dict[str, dict] = {}  # batchId -> batch info
_task_store: dict[str, dict] = {}  # taskId -> task info


# ============================================================
# 1. Validate API - 이미지 검증
# ============================================================


async def validate_images(request: ValidateRequest) -> ValidateResponse:
    """
    이미지 검증 메인 함수 (비동기)

    검증 순서:
    1. 포맷 검증 (로컬)
    2. 파일 크기 검증 (로컬)
    3. RunPod 호출 (NSFW, 패션, 임베딩)
    4. 중복/유사 이미지 검증 (Qdrant)
    5. 품질 검증 (로컬)
    """
    results: list[ValidationResult] = []
    passed_count = 0
    failed_count = 0

    # Step 1-2: 로컬 사전 검증 (포맷, 크기)
    pre_validated_urls = []
    pre_validation_errors = {}  # url -> error

    for image_url in request.images:
        # 포맷 검증 (jpg, png만 허용)
        if not _check_format(image_url):
            pre_validation_errors[image_url] = ValidationErrorCode.INVALID_FORMAT
            continue

        # 파일 크기 검증 (10MB 이하)
        file_size = _get_file_size(image_url)
        if file_size is None or file_size > MAX_FILE_SIZE_BYTES:
            pre_validation_errors[image_url] = ValidationErrorCode.FILE_TOO_LARGE
            continue

        pre_validated_urls.append(image_url)

    # Step 3: RunPod 호출 (NSFW, 패션, 임베딩)
    runpod_results = {}
    print(f"[DEBUG] pre_validated_urls: {len(pre_validated_urls)}개")
    if pre_validated_urls:
        print(f"[DEBUG] RunPod 호출 시작: {len(pre_validated_urls)}개 URL")
        ai_results = await call_runpod_validate(pre_validated_urls)
        print(f"[DEBUG] RunPod 호출 완료: {len(ai_results)}개 결과")
        for ai_result in ai_results:
            print(f"[DEBUG] AI Result: {ai_result}")  # 상세 결과 출력
            runpod_results[ai_result["url"]] = ai_result

    # Step 4-5: 최종 결과 생성
    for image_url in request.images:
        # 사전 검증 실패
        if image_url in pre_validation_errors:
            results.append(
                ValidationResult(
                    originUrl=image_url,
                    passed=False,
                    error=pre_validation_errors[image_url],
                )
            )
            failed_count += 1
            continue

        # RunPod 결과 확인
        ai_result = runpod_results.get(image_url, {})

        # 0. RunPod 에러 체크 (다운로드 실패 등)
        if ai_result.get("error"):
            logger.error(f"RunPod 에러: {ai_result.get('error')}")
            results.append(
                ValidationResult(
                    originUrl=image_url,
                    passed=False,
                    error=ValidationErrorCode.TOO_BLURRY,  # 임시로 품질 에러로 처리 (또는 새 코드 추가)
                )
            )
            failed_count += 1
            continue

        # NSFW 검증
        nsfw_info = ai_result.get("nsfw", {})
        if nsfw_info.get("is_nsfw", False):
            results.append(
                ValidationResult(
                    originUrl=image_url,
                    passed=False,
                    error=ValidationErrorCode.NSFW,
                )
            )
            failed_count += 1
            continue

        # 패션 도메인 검증
        fashion_info = ai_result.get("fashion", {})
        if not fashion_info.get("is_fashion", True):
            results.append(
                ValidationResult(
                    originUrl=image_url,
                    passed=False,
                    error=ValidationErrorCode.NOT_FASHION,
                )
            )
            failed_count += 1
            continue

        # 중복/유사 이미지 검증
        embedding = ai_result.get("embedding", [])

        # 중복 검사 (Qdrant)
        similarity = _check_duplicate(image_url, request.userId, embedding)
        if similarity >= SIMILARITY_THRESHOLD:
            results.append(
                ValidationResult(
                    originUrl=image_url,
                    passed=False,
                    error=ValidationErrorCode.DUPLICATE,
                    embedding=embedding,  # 중복일 때도 임베딩 반환 (디버깅용)
                )
            )
            failed_count += 1
            continue

        # 품질 검증
        if not _check_quality(image_url):
            results.append(
                ValidationResult(
                    originUrl=image_url,
                    passed=False,
                    error=ValidationErrorCode.TOO_BLURRY,
                )
            )
            failed_count += 1
            continue

        # 모든 검증 통과
        results.append(
            ValidationResult(
                originUrl=image_url,
                passed=True,
                embedding=embedding,  # 중요: 클라이언트가 저장할 수 있도록 임베딩 반환
            )
        )
        passed_count += 1

    return ValidateResponse(
        success=True,
        validationSummary=ValidationSummary(
            total=len(request.images), passed=passed_count, failed=failed_count
        ),
        validationResults=results,
    )


# ============================================================
# 검증 헬퍼 함수들
# ============================================================


def _check_format(image_url: str) -> bool:
    """이미지 포맷 검증 (확장자 기반)"""
    # URL에서 확장자 추출
    url_path = image_url.split("?")[0]  # 쿼리스트링 제거

    # 파일명 추출 (경로의 마지막 부분)
    filename = url_path.split("/")[-1]

    # 파일명에 확장자가 있는 경우만 체크
    if "." in filename:
        extension = filename.rsplit(".", 1)[-1].lower()
        result = extension in ALLOWED_FORMATS
        logger.info(f"포맷 체크: {filename}, 확장자: {extension}, 결과: {result}")
        return result

    # 확장자가 없으면 일단 통과 (RunPod에서 다운로드 시도)
    logger.info(f"포맷 체크: {filename}, 확장자 없음, 통과")
    return True


def _get_file_size(image_url: str) -> Optional[int]:
    """파일 크기 조회 (HEAD 요청)"""
    try:
        with httpx.Client() as client:
            response = client.head(image_url, timeout=10)
            content_length = response.headers.get("content-length")
            return int(content_length) if content_length else None
    except Exception:
        return None


# Qdrant 서비스 초기화
# qdrant_service = QdrantService() <--- Global variable initialized at module level (moved down)

qdrant_service = QdrantService()


def _check_duplicate(image_url: str, user_id: int, embedding: list[float]) -> float:
    """
    중복/유사 이미지 검증

    RunPod에서 생성된 임베딩을 Qdrant에서 검색하여 유사 이미지 확인

    Args:
        image_url: 이미지 URL
        user_id: 사용자 ID
        embedding: RunPod에서 생성된 임베딩 벡터

    Returns:
        float: 최대 유사도 (0.0 ~ 1.0)
    """
    if not embedding:
        return 0.0

    # Qdrant 유사 검색
    results = qdrant_service.search_similar(
        vector=embedding, user_id=user_id, limit=1, score_threshold=0.0
    )

    if results:
        best_match = results[0]
        logger.info(f"유사 이미지 발견: ID {best_match.id}, 점수 {best_match.score}")
        return float(best_match.score)

    return 0.0


def save_embedding(
    user_id: int, clothes_id: int, embedding: list[float], metadata: dict
) -> bool:
    """
    임베딩 저장 (Qdrant) - 내부 전용

    Note: analyze API 완료 시 자동 호출됨 (외부 API 아님)

    Args:
        user_id: 사용자 ID
        clothes_id: 옷 ID
        embedding: 임베딩 벡터
        metadata: 메타데이터 (category, color, material)

    Returns:
        성공 여부
    """
    if not embedding:
        return False

    payload = {"user_id": user_id, **metadata}

    return qdrant_service.upsert_item(id=clothes_id, vector=embedding, payload=payload)


def delete_item_from_qdrant(clothes_id: int) -> bool:
    """
    아이템 삭제 (Qdrant)

    Args:
        clothes_id: 삭제할 옷 ID

    Returns:
        성공 여부
    """
    return qdrant_service.delete_item(clothes_id)


def _check_quality(image_url: str) -> bool:
    """
    이미지 품질 검증 (해상도)

    최소 해상도: 512x512 픽셀

    Note: 블러 검출은 구현 안 함
    - 고도화 시 고려: Super Resolution (Real-ESRGAN 등)으로 흐릿한 사진 선명화
    """
    try:
        import io

        from PIL import Image

        # 이미지 다운로드
        with httpx.Client(timeout=10, verify=False) as client:
            response = client.get(image_url)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            width, height = image.size

            # 최소 해상도 체크
            if width < MIN_IMAGE_RESOLUTION or height < MIN_IMAGE_RESOLUTION:
                logger.info(
                    f"해상도 부족: {width}x{height} (최소: {MIN_IMAGE_RESOLUTION}x{MIN_IMAGE_RESOLUTION})"
                )
                return False

            logger.info(f"품질 양호: {width}x{height}")
            return True

    except Exception as e:
        logger.error(f"품질 검증 실패: {image_url}, 에러: {e}")
        return True  # 에러 시 통과 (너그럽게)


# ============================================================
# RunPod API 호출
# ============================================================


async def call_runpod_validate(image_urls: list[str]) -> list[dict]:
    """
    RunPod Serverless로 이미지 검증 요청

    Args:
        image_urls: 검증할 이미지 URL 목록

    Returns:
        검증 결과 목록
        [
            {
                "url": "...",
                "nsfw": {"is_nsfw": false, "score": 0.1},
                "fashion": {"is_fashion": true, "score": 0.8},
                "embedding": [0.1, 0.2, ...]
            }
        ]
    """
    logger.info(f"USE_MOCK_VALIDATOR: {USE_MOCK_VALIDATOR}")
    if USE_MOCK_VALIDATOR:
        logger.info("Mock 모드로 실행")
        return _mock_validate(image_urls)

    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        logger.warning("RunPod 설정이 없습니다. Mock 모드로 실행합니다.")
        return _mock_validate(image_urls)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{RUNPOD_BASE_URL}/runsync",
                headers={
                    "Authorization": f"Bearer {RUNPOD_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": {
                        "action": "validate",
                        "images": image_urls,
                    }
                },
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "COMPLETED":
                return data.get("output", {}).get("results", [])
            else:
                logger.error(f"RunPod 작업 실패: {data}")
                return _mock_validate(image_urls)

    except Exception as e:
        logger.error(f"RunPod 호출 실패: {e}")
        return _mock_validate(image_urls)


def _mock_validate(image_urls: list[str]) -> list[dict]:
    """
    Mock 검증 결과 (테스트용)
    """
    results = []
    for url in image_urls:
        # URL에 특정 키워드가 있으면 테스트용으로 실패 처리
        is_nsfw = "nsfw" in url.lower()
        is_fashion = "food" not in url.lower() and "landscape" not in url.lower()

        results.append(
            {
                "url": url,
                "nsfw": {"is_nsfw": is_nsfw, "score": 0.9 if is_nsfw else 0.05},
                "fashion": {
                    "is_fashion": is_fashion,
                    "score": 0.85 if is_fashion else 0.15,
                },
                "embedding": [0.1] * 768 if is_fashion and not is_nsfw else [],
            }
        )
    return results


# ============================================================
# 2. Analyze API - 이미지 분석 시작
# ============================================================


def start_analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    이미지 분석 작업 시작 (비동기)

    작업 내용:
    1. 배경 제거 (rembg)
    2. AI 속성 분석 (카테고리, 색상, 소재, 스타일)
    """
    batch_id = request.batchId

    # 배치 정보 저장
    _batch_store[batch_id] = {
        "user_id": request.userId,
        "status": BatchStatus.ACCEPTED,
        "total": len(request.images),
        "completed": 0,
        "processing": len(request.images),
        "created_at": time.time(),
    }

    # 개별 작업 정보 저장
    for image in request.images:
        task_id = image.taskId
        _task_store[task_id] = {
            "batch_id": batch_id,
            "user_id": request.userId,
            "sequence": image.sequence,
            "target_image": image.targetImage,
            "file_info": {
                "file_id": image.fileUploadInfo.fileId,
                "object_key": image.fileUploadInfo.objectKey,
                "presigned_url": image.fileUploadInfo.presignedUrl,
            },
            "status": TaskStatus.PREPROCESSING,
            "file_id": None,
            "analysis_result": None,
            "created_at": time.time(),
        }

        # TODO: 백그라운드 작업 큐에 추가 (Celery, asyncio 등)
        # _enqueue_task(task_id)

    return AnalyzeResponse(
        batchId=batch_id,
        status=BatchStatus.ACCEPTED,
        meta=BatchMeta(
            total=len(request.images),
            completed=0,
            processing=len(request.images),
            isFinished=False,
        ),
    )


# ============================================================
# 3. Batch Status API - 작업 상태 조회
# ============================================================


def get_batch_status(batch_id: str) -> Optional[BatchStatusResponse]:
    """배치 상태 조회"""

    if batch_id not in _batch_store:
        return None

    batch = _batch_store[batch_id]

    # 해당 배치의 모든 작업 조회
    results: list[BatchResultItem] = []
    completed_count = 0
    processing_count = 0

    for task_id, task in _task_store.items():
        if task.get("batch_id") != batch_id:
            continue

        result_item = _build_batch_result_item(task_id, task)
        results.append(result_item)

        if task["status"] == TaskStatus.COMPLETED:
            completed_count += 1
        elif task["status"] != TaskStatus.FAILED:
            processing_count += 1

    # 결과를 sequence 순으로 정렬
    results.sort(key=lambda x: _task_store.get(x.taskId, {}).get("sequence", 0))

    total = batch["total"]
    is_finished = completed_count == total

    # 전체 상태 결정
    if is_finished:
        overall_status = BatchStatus.COMPLETED
    else:
        overall_status = BatchStatus.IN_PROGRESS

    return BatchStatusResponse(
        batchId=batch_id,
        status=overall_status,
        meta=BatchMeta(
            total=total,
            completed=completed_count,
            processing=processing_count,
            isFinished=is_finished,
        ),
        results=results,
    )


def _build_batch_result_item(task_id: str, task: dict) -> BatchResultItem:
    """BatchResultItem 빌드"""

    attributes = None
    if task.get("analysis_result"):
        attrs = task["analysis_result"]
        attributes = AnalysisAttributes(
            category=attrs.get("category", ""),
            color=attrs.get("color", []),
            material=attrs.get("material", []),
            styleTags=attrs.get("style_tags", []),
        )

    return BatchResultItem(
        taskId=task_id,
        status=task["status"],
        fileId=task.get("file_id"),
        attributes=attributes,
    )


# ============================================================
# 분석 작업 실행 함수들 (백그라운드에서 호출)
# ============================================================


async def process_image_task(task_id: str) -> None:
    """
    이미지 분석 작업 실행 (백그라운드)

    1. 배경 제거
    2. AI 분석
    """
    if task_id not in _task_store:
        return

    task = _task_store[task_id]

    try:
        # Step 1: 배경 제거 (PREPROCESSING)
        task["status"] = TaskStatus.PREPROCESSING

        processed_file_id = await _remove_background(
            task["file_info"]["presigned_url"], task["file_info"]["file_id"]
        )

        task["file_id"] = processed_file_id

        # Step 2: AI 분석 (ANALYZING)
        task["status"] = TaskStatus.ANALYZING

        attributes = await _analyze_image_attributes(processed_file_id)

        task["analysis_result"] = attributes

        # 완료
        task["status"] = TaskStatus.COMPLETED

        # 배치 완료 카운트 업데이트
        batch_id = task["batch_id"]
        if batch_id in _batch_store:
            _batch_store[batch_id]["completed"] += 1
            _batch_store[batch_id]["processing"] -= 1

    except Exception as e:
        task["status"] = TaskStatus.FAILED
        task["error"] = str(e)


async def _remove_background(presigned_url: str, file_id: int) -> int:
    """
    배경 제거 (rembg)

    Returns:
        처리된 이미지의 파일 ID
    """
    # TODO: rembg 또는 GPU 워커(RunPod) 호출
    # 1. presigned_url에서 이미지 다운로드
    # 2. 배경 제거
    # 3. S3 업로드
    # 4. 새 file_id 반환
    return file_id  # 임시: 같은 file_id 반환


async def _analyze_image_attributes(file_id: int) -> dict:
    """
    AI 이미지 분석 (BLIP-2 등)

    Returns:
        분석된 속성들
    """
    # TODO: BLIP-2 또는 GPU 워커 호출
    # 1. 이미지 로드
    # 2. 카테고리 분류
    # 3. 색상 추출
    # 4. 소재 분석
    # 5. 스타일 태그 생성
    return {
        "category": "상의",
        "color": ["검정"],
        "material": ["면"],
        "style_tags": ["캐주얼", "베이직"],
    }  # 임시
