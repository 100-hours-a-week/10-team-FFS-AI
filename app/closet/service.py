"""
Closet 서비스 - 이미지 검증 및 분석 비즈니스 로직 (API 명세 v2 기준)

주요 기능:
1. 이미지 검증 (Validate): 포맷, 크기, 중복/유사(Marqo-FashionSigLIP), 패션 여부(LAION CLIP), NSFW(WD14)
2. 이미지 분석 (Analyze): 배경 제거, AI 속성 분석
3. 작업 상태 관리: 비동기 작업 진행 상태 추적

AI 모델:
- Marqo-FashionSigLIP: 중복/유사 이미지 검증 (임베딩 기반)
- LAION CLIP-ViT-B-32-laion2b-s34b-b79k: 도메인 탐지 (패션 여부)
- WD14 Tagger: NSFW 검증
"""

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

# ============================================================
# 설정 상수
# ============================================================

ALLOWED_FORMATS = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MIN_IMAGE_SIZE = 100  # 최소 100x100
SIMILARITY_THRESHOLD = 0.95  # 유사도 임계값


# ============================================================
# 작업 저장소 (실제로는 Redis 등 사용)
# ============================================================

# 임시 인메모리 저장소 (나중에 Redis로 교체)
_batch_store: dict[str, dict] = {}  # batchId -> batch info
_task_store: dict[str, dict] = {}  # taskId -> task info


# ============================================================
# 1. Validate API - 이미지 검증
# ============================================================


def validate_images(request: ValidateRequest) -> ValidateResponse:
    """
    이미지 검증 메인 함수

    검증 순서:
    1. 포맷 검증
    2. 파일 크기 검증
    3. 중복/유사 이미지 검증 (Marqo-FashionSigLIP)
    4. 패션 도메인 검증 (LAION CLIP-ViT-B-32-laion2b-s34b-b79k)
    5. NSFW 검증 (WD14 Tagger)
    6. 품질 검증 (라플라시안)
    """
    results: list[ValidationResult] = []
    passed_count = 0
    failed_count = 0

    for image_url in request.images:
        result = _validate_single_image(image_url, request.userId)
        results.append(result)

        if result.passed:
            passed_count += 1
        else:
            failed_count += 1

    return ValidateResponse(
        success=True,
        validationSummary=ValidationSummary(
            total=len(request.images), passed=passed_count, failed=failed_count
        ),
        validationResults=results,
    )


def _validate_single_image(image_url: str, user_id: int) -> ValidationResult:
    """개별 이미지 검증"""

    # Step 1: 포맷 검증
    if not _check_format(image_url):
        return ValidationResult(
            originUrl=image_url,
            passed=False,
            error=ValidationErrorCode.INVALID_FORMAT,
        )

    # Step 2: 파일 크기 검증
    file_size = _get_file_size(image_url)
    if file_size is None or file_size > MAX_FILE_SIZE_BYTES:
        return ValidationResult(
            originUrl=image_url,
            passed=False,
            error=ValidationErrorCode.FILE_TOO_LARGE,
        )

    # Step 3: 중복/유사 이미지 검증 (Marqo-FashionSigLIP)
    similarity = _check_duplicate(image_url, user_id)
    if similarity >= SIMILARITY_THRESHOLD:
        return ValidationResult(
            originUrl=image_url, passed=False, error=ValidationErrorCode.DUPLICATE
        )

    # Step 4: 패션 도메인 검증 (LAION CLIP)
    if not _check_is_fashion(image_url):
        return ValidationResult(
            originUrl=image_url, passed=False, error=ValidationErrorCode.NOT_FASHION
        )

    # Step 5: NSFW 검증 (WD14 Tagger)
    wd14_result = _check_with_wd14_tagger(image_url)
    if wd14_result["is_nsfw"]:
        return ValidationResult(
            originUrl=image_url, passed=False, error=ValidationErrorCode.NSFW
        )

    # Step 6: 품질 검증
    if not _check_quality(image_url):
        return ValidationResult(
            originUrl=image_url, passed=False, error=ValidationErrorCode.TOO_BLURRY
        )

    # 모든 검증 통과
    return ValidationResult(originUrl=image_url, passed=True)


# ============================================================
# 검증 헬퍼 함수들
# ============================================================


def _check_format(image_url: str) -> bool:
    """이미지 포맷 검증 (확장자 기반)"""
    # URL에서 확장자 추출
    url_path = image_url.split("?")[0]  # 쿼리스트링 제거
    extension = url_path.rsplit(".", 1)[-1].lower()
    return extension in ALLOWED_FORMATS


def _get_file_size(image_url: str) -> Optional[int]:
    """파일 크기 조회 (HEAD 요청)"""
    try:
        with httpx.Client() as client:
            response = client.head(image_url, timeout=10)
            content_length = response.headers.get("content-length")
            return int(content_length) if content_length else None
    except Exception:
        return None


def _check_duplicate(image_url: str, user_id: int) -> float:
    """
    중복/유사 이미지 검증 (Marqo-FashionSigLIP 기반)

    Marqo-FashionSigLIP 모델로 이미지 임베딩 생성 후
    Qdrant에서 유사 이미지 검색

    Returns:
        float: 최대 유사도 (0.0 ~ 1.0)
        - similarity >= 0.53: 중복으로 판단 (DUPLICATE)
    """
    # TODO: Marqo-FashionSigLIP 연동
    # 1. 이미지 다운로드
    # 2. Marqo-FashionSigLIP으로 임베딩 생성
    # 3. Qdrant에서 user_id 필터로 유사 검색
    # 4. 최대 유사도 반환

    # 임시 구현
    return 0.0


def _check_with_wd14_tagger(image_url: str) -> dict:
    """
    NSFW 검증 (WD14 Tagger)

    WD14 Tagger는 이미지에서 태그를 추출하여:
    - NSFW 관련 태그로 부적절 콘텐츠 감지

    Note: 패션 여부는 LAION CLIP으로 별도 처리

    Returns:
        {
            "is_nsfw": bool,     # NSFW인지
            "tags": list[str],   # 감지된 태그들
            "confidence": float  # 신뢰도
        }
    """
    # TODO: WD14 Tagger 연동
    # 1. 이미지 다운로드 및 전처리
    # 2. WD14 Tagger 모델로 태그 추출
    # 3. NSFW 태그 확인 (rating: questionable, explicit 등)

    # 임시 구현
    return {"is_nsfw": False, "tags": [], "confidence": 0.0}


def _check_is_fashion(image_url: str) -> bool:
    """
    패션 도메인 탐지 (LAION CLIP-ViT-B-32-laion2b-s34b-b79k)

    LAION CLIP 모델을 사용하여 이미지가 패션 도메인인지 판별
    - 정확도와 VRAM 효율성 측면에서 최적의 성능

    Returns:
        bool: 패션 아이템인지 여부
    """
    # TODO: LAION CLIP 연동
    # 1. 이미지 다운로드 및 전처리
    # 2. LAION CLIP-ViT-B-32-laion2b-s34b-b79k 모델로 분류
    # 3. 패션 도메인 여부 반환

    # 임시 구현
    return True


def _check_quality(image_url: str) -> bool:
    """이미지 품질 검증 (블러, 해상도 등)"""
    # TODO: 라플라시안 분산으로 블러 검출
    # 1. 이미지 로드
    # 2. 블러 점수 계산
    # 3. 해상도 확인
    return True  # 임시: 품질 양호


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
