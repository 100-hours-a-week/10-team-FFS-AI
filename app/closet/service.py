"""
Closet 서비스 - 이미지 검증 및 분석 비즈니스 로직

주요 기능:
1. 이미지 검증 (Validate): 포맷, 크기, 중복/유사(Marqo-FashionCLIP), NSFW/패션(WD14 Tagger)
2. 이미지 분석 (Analyze): 배경 제거, AI 속성 분석
3. 작업 상태 관리: 비동기 작업 진행 상태 추적

AI 모델:
- Marqo-FashionCLIP (B/16): 중복/유사 이미지 검증 (임베딩 기반)
- WD14 Tagger: 패션 여부 + NSFW 검증
"""

import time
import hashlib
import uuid
from typing import Optional
from PIL import Image
import httpx

from app.closet.schemas import (
    ValidateRequest,
    ValidateResponse,
    ValidationResult,
    ValidationSummary,
    ValidationStatus,
    ValidationErrorCode,
    AnalyzeRequest,
    AnalyzeResponse,
    TaskIdItem,
    TaskStatus,
    TaskStatusResponse,
    TaskMeta,
    TaskResultItem,
    ProgressInfo,
    ProgressStatus,
    BackgroundRemovalResult,
    AnalysisResult,
    AnalysisAttributes,
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
_task_store: dict[str, dict] = {}


# ============================================================
# 1. Validate API - 이미지 검증
# ============================================================

def validate_images(request: ValidateRequest) -> ValidateResponse:
    """
    이미지 검증 메인 함수
    
    검증 순서:
    1. 포맷 검증
    2. 파일 크기 검증
    3. 중복/유사 이미지 검증 (Marqo-FashionCLIP)
    4. 패션 아이템 여부 (WD14 Tagger)
    5. NSFW 검증 (WD14 Tagger)
    6. 품질 검증 (라플라시안)
    """
    start_time = time.time()
    results: list[ValidationResult] = []
    passed_count = 0
    failed_count = 0
    
    for image_url in request.images:
        result = _validate_single_image(image_url, request.userId)
        results.append(result)
        
        if result.status == ValidationStatus.PASSED:
            passed_count += 1
        else:
            failed_count += 1
    
    processing_time_ms = int((time.time() - start_time) * 1000)
    
    return ValidateResponse(
        success=True,
        validationId=f"val_{_generate_ulid()}",
        processingTimeMs=processing_time_ms,
        validationSummary=ValidationSummary(
            total=len(request.images),
            passed=passed_count,
            failed=failed_count
        ),
        validationResults=results
    )


def _validate_single_image(image_url: str, user_id: str) -> ValidationResult:
    """개별 이미지 검증"""
    
    # Step 1: 포맷 검증
    if not _check_format(image_url):
        return ValidationResult(
            originUrl=image_url,
            status=ValidationStatus.FAILED,
            errorCode=ValidationErrorCode.INVALID_FORMAT
        )
    
    # Step 2: 파일 크기 검증
    file_size = _get_file_size(image_url)
    if file_size is None or file_size > MAX_FILE_SIZE_BYTES:
        return ValidationResult(
            originUrl=image_url,
            status=ValidationStatus.FAILED,
            errorCode=ValidationErrorCode.FILE_TOO_LARGE
        )
    
    # Step 3: 중복/유사 이미지 검증 (Marqo-FashionCLIP)
    similarity, is_exact = _check_duplicate_or_similar(image_url, user_id)
    if is_exact:
        return ValidationResult(
            originUrl=image_url,
            status=ValidationStatus.FAILED,
            errorCode=ValidationErrorCode.EXACT_DUPLICATE
        )
    if similarity >= SIMILARITY_THRESHOLD:
        return ValidationResult(
            originUrl=image_url,
            status=ValidationStatus.FAILED,
            errorCode=ValidationErrorCode.SIMILAR_ITEM
        )
    
    # Step 4-5: 패션 여부 + NSFW 검증 (WD14 Tagger)
    wd14_result = _check_with_wd14_tagger(image_url)
    
    if not wd14_result["is_fashion"]:
        return ValidationResult(
            originUrl=image_url,
            status=ValidationStatus.FAILED,
            errorCode=ValidationErrorCode.NOT_FASHION
        )
    
    if wd14_result["is_nsfw"]:
        return ValidationResult(
            originUrl=image_url,
            status=ValidationStatus.FAILED,
            errorCode=ValidationErrorCode.NSFW_DETECTED
        )
    
    # Step 6: 품질 검증
    if not _check_quality(image_url):
        return ValidationResult(
            originUrl=image_url,
            status=ValidationStatus.FAILED,
            errorCode=ValidationErrorCode.TOO_BLURRY
        )
    
    # 모든 검증 통과
    return ValidationResult(
        originUrl=image_url,
        status=ValidationStatus.PASSED
    )


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


def _check_duplicate_or_similar(image_url: str, user_id: str) -> tuple[float, bool]:
    """
    중복/유사 이미지 검증 (Marqo-FashionCLIP 기반)
    
    Marqo-FashionCLIP (B/16) 모델로 이미지 임베딩 생성 후
    Qdrant에서 유사 이미지 검색
    
    Returns:
        (similarity, is_exact): 유사도와 완전 동일 여부
        - similarity >= 0.99: 완전 동일 (EXACT_DUPLICATE)
        - similarity >= 0.95: 유사 아이템 (SIMILAR_ITEM)
    """
    # TODO: Marqo-FashionCLIP 연동
    # 1. 이미지 다운로드
    # 2. Marqo-FashionCLIP으로 임베딩 생성
    # 3. Qdrant에서 user_id 필터로 유사 검색
    # 4. 최대 유사도 반환
    
    # 임시 구현
    similarity = 0.0
    is_exact = similarity >= 0.99
    return similarity, is_exact


def _check_with_wd14_tagger(image_url: str) -> dict:
    """
    WD14 Tagger로 패션 여부 + NSFW 검증
    
    WD14 Tagger는 이미지에서 태그를 추출하여:
    - 패션 관련 태그 존재 여부로 패션 아이템 판별
    - NSFW 관련 태그로 부적절 콘텐츠 감지
    
    Returns:
        {
            "is_fashion": bool,  # 패션 아이템인지
            "is_nsfw": bool,     # NSFW인지
            "tags": list[str],   # 감지된 태그들
            "confidence": float  # 신뢰도
        }
    """
    # TODO: WD14 Tagger 연동
    # 1. 이미지 다운로드 및 전처리
    # 2. WD14 Tagger 모델로 태그 추출
    # 3. 패션 태그 확인 (shirt, pants, dress, jacket 등)
    # 4. NSFW 태그 확인 (nude, explicit 등)
    
    # 임시 구현
    return {
        "is_fashion": True,
        "is_nsfw": False,
        "tags": [],
        "confidence": 0.0
    }


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
    main_task_id = _generate_ulid()
    task_ids: list[TaskIdItem] = []
    
    for image in request.images:
        sub_task_id = _generate_ulid()
        
        # 작업 상태 초기화
        _task_store[sub_task_id] = {
            "main_task_id": main_task_id,
            "user_id": request.userId,
            "origin_url": image.originUrl,
            "upload_url": image.uploadUrl,
            "image_key": image.imageKey,
            "sequence": image.sequence,
            "status": TaskStatus.PROCESSING,
            "progress": {
                "background_removal": ProgressStatus.PENDING,
                "analysis": ProgressStatus.PENDING
            },
            "background_removal_result": None,
            "analysis_result": None,
            "created_at": time.time()
        }
        
        task_ids.append(TaskIdItem(
            originUrl=image.originUrl,
            taskId=sub_task_id
        ))
        
        # TODO: 백그라운드 작업 큐에 추가 (Celery, asyncio 등)
        # _enqueue_task(sub_task_id)
    
    # 메인 작업 정보 저장
    _task_store[main_task_id] = {
        "type": "main",
        "user_id": request.userId,
        "sub_task_ids": [t.taskId for t in task_ids],
        "status": TaskStatus.PROCESSING,
        "created_at": time.time()
    }
    
    return AnalyzeResponse(
        taskId=main_task_id,
        status=TaskStatus.PROCESSING,
        queued=len(request.images),
        taskIds=task_ids
    )


# ============================================================
# 3. Task Status API - 작업 상태 조회
# ============================================================

def get_task_status(task_id: str) -> Optional[TaskStatusResponse]:
    """작업 상태 조회"""
    
    if task_id not in _task_store:
        return None
    
    task = _task_store[task_id]
    
    # 메인 작업인 경우
    if task.get("type") == "main":
        return _get_main_task_status(task_id)
    
    # 개별 작업인 경우 (해당 작업만 반환)
    return _get_single_task_status(task_id)


def _get_main_task_status(main_task_id: str) -> TaskStatusResponse:
    """메인 작업 상태 조회 (모든 하위 작업 포함)"""
    main_task = _task_store[main_task_id]
    sub_task_ids = main_task["sub_task_ids"]
    
    results: list[TaskResultItem] = []
    completed_count = 0
    processing_count = 0
    
    for sub_id in sub_task_ids:
        sub_task = _task_store.get(sub_id)
        if not sub_task:
            continue
        
        result_item = _build_task_result_item(sub_id, sub_task)
        results.append(result_item)
        
        if sub_task["status"] == TaskStatus.COMPLETED:
            completed_count += 1
        else:
            processing_count += 1
    
    total = len(sub_task_ids)
    is_finished = completed_count == total
    
    overall_status = TaskStatus.COMPLETED if is_finished else TaskStatus.PROCESSING
    
    return TaskStatusResponse(
        taskId=main_task_id,
        status=overall_status,
        meta=TaskMeta(
            total=total,
            completed=completed_count,
            processing=processing_count,
            isFinished=is_finished
        ),
        results=results
    )


def _get_single_task_status(task_id: str) -> TaskStatusResponse:
    """개별 작업 상태 조회"""
    task = _task_store[task_id]
    main_task_id = task.get("main_task_id", task_id)
    
    result_item = _build_task_result_item(task_id, task)
    
    return TaskStatusResponse(
        taskId=main_task_id,
        status=task["status"],
        meta=TaskMeta(
            total=1,
            completed=1 if task["status"] == TaskStatus.COMPLETED else 0,
            processing=0 if task["status"] == TaskStatus.COMPLETED else 1,
            isFinished=task["status"] == TaskStatus.COMPLETED
        ),
        results=[result_item]
    )


def _build_task_result_item(task_id: str, task: dict) -> TaskResultItem:
    """TaskResultItem 빌드"""
    progress = task.get("progress", {})
    
    bg_removal_result = None
    if task.get("background_removal_result"):
        bg_removal_result = BackgroundRemovalResult(
            imageKey=task["background_removal_result"]["image_key"]
        )
    
    analysis_result = None
    if task.get("analysis_result"):
        attrs = task["analysis_result"]
        analysis_result = AnalysisResult(
            attributes=AnalysisAttributes(
                category=attrs.get("category", ""),
                color=attrs.get("color", []),
                material=attrs.get("material", []),
                styleTags=attrs.get("style_tags", [])
            )
        )
    
    return TaskResultItem(
        taskId=task_id,
        originUrl=task["origin_url"],
        status=task["status"],
        progress=ProgressInfo(
            backgroundRemoval=progress.get("background_removal", ProgressStatus.PENDING),
            analysis=progress.get("analysis", ProgressStatus.PENDING)
        ),
        backgroundRemoval=bg_removal_result,
        analysis=analysis_result
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
        # Step 1: 배경 제거
        task["progress"]["background_removal"] = ProgressStatus.IN_PROGRESS
        
        processed_image_key = await _remove_background(task["upload_url"])
        
        task["progress"]["background_removal"] = ProgressStatus.DONE
        task["background_removal_result"] = {"image_key": processed_image_key}
        
        # Step 2: AI 분석
        task["progress"]["analysis"] = ProgressStatus.IN_PROGRESS
        
        attributes = await _analyze_image_attributes(processed_image_key)
        
        task["progress"]["analysis"] = ProgressStatus.DONE
        task["analysis_result"] = attributes
        
        # 완료
        task["status"] = TaskStatus.COMPLETED
        
    except Exception as e:
        task["status"] = TaskStatus.FAILED
        task["error"] = str(e)


async def _remove_background(image_url: str) -> str:
    """
    배경 제거 (rembg)
    
    Returns:
        처리된 이미지의 S3 키
    """
    # TODO: rembg 또는 GPU 워커(RunPod) 호출
    # 1. 이미지 다운로드
    # 2. 배경 제거
    # 3. S3 업로드
    # 4. 새 이미지 키 반환
    return "processed/bg_removed_image.png"  # 임시


async def _analyze_image_attributes(image_key: str) -> dict:
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
        "style_tags": ["캐주얼", "베이직"]
    }  # 임시


# ============================================================
# 유틸리티 함수
# ============================================================

def _generate_ulid() -> str:
    """ULID 생성 (임시: UUID4 사용)"""
    # TODO: 실제 ULID 라이브러리 사용
    return uuid.uuid4().hex[:26].upper()
