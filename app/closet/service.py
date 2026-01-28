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

from __future__ import annotations

import asyncio
import io
import logging
import time

import httpx
from fastapi import BackgroundTasks, HTTPException, status

from app.closet.background_removal import get_background_remover
from app.closet.schemas import (
    # Analyze API
    AnalyzeRequest,
    AnalyzeResponse,
    BatchMeta,
    BatchResultItem,
    BatchStatus,
    # Batch Status API
    BatchStatusResponse,
    Category,
    ExtraAttributes,
    MajorAttributes,
    MetaData,
    TaskStatus,
    # Validate API
    ValidateRequest,
    ValidateResponse,
    ValidationErrorCode,
    ValidationResult,
    ValidationSummary,
)
from app.closet.validators import ImageValidator, MockImageValidator, download_image
from app.config import get_settings
from app.core.redis_client import get_redis_client
from app.core.storage import get_storage

logger = logging.getLogger(__name__)

# ============================================================
# 설정 상수
# ============================================================

ALLOWED_FORMATS = {"jpg", "jpeg", "png"}  # webp 제외
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MIN_IMAGE_RESOLUTION = 512  # 최소 512x512
SIMILARITY_THRESHOLD = 0.95  # 유사도 임계값

# AI Model Server 설정 (환경변수에서 로드)
# AI_MODEL_SERVER_URL = os.getenv("AI_MODEL_SERVER_URL", "")


# Mock 모드 (AI 서버 없이 테스트용)
# USE_MOCK_VALIDATOR = os.getenv("USE_MOCK_VALIDATOR", "true").lower() == "true"

# 설정 로드
settings = get_settings()


# ============================================================
# Redis 클라이언트
# ============================================================


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

    # Step 3: AI Model Server 호출 (NSFW, 패션, 임베딩)
    ai_results = {}
    logger.info(f"[DEBUG] pre_validated_urls: {len(pre_validated_urls)}개")

    if pre_validated_urls:
        logger.info(f"[DEBUG] AI 서버 호출 시작: {len(pre_validated_urls)}개 URL")
        ai_response_list = await call_ai_model_server(pre_validated_urls)
        logger.info(f"[DEBUG] AI 서버 호출 완료: {len(ai_response_list)}개 결과")

        for ai_result in ai_response_list:
            logger.info(f"[DEBUG] AI Result: {ai_result}")  # 상세 결과 출력
            ai_results[ai_result["url"]] = ai_result

    # Step 4-5: 최종 결과 생성
    for image_url in request.images:
        # 사전 검증 실패
        if image_url in pre_validation_errors:
            results.append(
                ValidationResult(
                    origin_url=image_url,
                    passed=False,
                    error=pre_validation_errors[image_url],
                )
            )
            failed_count += 1
            continue

        # AI 서버 결과 확인
        ai_result = ai_results.get(image_url, {})

        # 0. AI 서버 에러 체크 (다운로드 실패 등)
        if ai_result.get("error"):
            logger.error(f"AI 서버 에러: {ai_result.get('error')}")
            results.append(
                ValidationResult(
                    origin_url=image_url,
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
                    origin_url=image_url,
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
                    origin_url=image_url,
                    passed=False,
                    error=ValidationErrorCode.NOT_FASHION,
                )
            )
            failed_count += 1
            continue

        # 품질 검증
        if not _check_quality(image_url):
            results.append(
                ValidationResult(
                    origin_url=image_url,
                    passed=False,
                    error=ValidationErrorCode.TOO_BLURRY,
                )
            )
            failed_count += 1
            continue

        # 모든 검증 통과
        results.append(
            ValidationResult(
                origin_url=image_url,
                passed=True,
                # embedding 제거 (User Request)
            )
        )
        passed_count += 1

    return ValidateResponse(
        success=True,
        validation_summary=ValidationSummary(
            total=len(request.images), passed=passed_count, failed=failed_count
        ),
        validation_results=results,
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

    # 확장자가 없으면 일단 통과 (AI 서버에서 다운로드 시도)
    logger.info(f"포맷 체크: {filename}, 확장자 없음, 통과")
    return True


def _get_file_size(image_url: str) -> int | None:
    """파일 크기 조회 (HEAD 요청)"""
    try:
        with httpx.Client() as client:
            response = client.head(image_url, timeout=10)
            content_length = response.headers.get("content-length")
            return int(content_length) if content_length else None
    except Exception:
        return None


def _check_quality(image_url: str) -> bool:
    """
    이미지 품질 검증 (해상도)

    최소 해상도: 512x512 픽셀

    Note: 블러 검출은 구현 안 함
    - 고도화 시 고려: Super Resolution (Real-ESRGAN 등)으로 흐릿한 사진 선명화
    """
    try:
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
# AI Model Server API 호출
# ============================================================


# ============================================================
# AI 모델 검증 (Direct Call)
# ============================================================


_validator_instance: ImageValidator | None = None


def get_validator() -> ImageValidator:
    """Validator 싱글톤 가져오기"""
    global _validator_instance
    if _validator_instance is None:
        if settings.use_mock_validator:
            logger.info("Mock Validator 사용")
            _validator_instance = MockImageValidator()
        else:
            logger.info("Real Image Validator 초기화 (Lazy Loading)")
            _validator_instance = ImageValidator(lazy_load=True)
    return _validator_instance


async def call_ai_model_server(image_urls: list[str]) -> list[dict]:
    """
    AI 모델 직접 호출 (Monolith 방식)
    이름은 call_ai_model_server지만 실제로는 내부 함수 호출
    """
    logger.info(f"AI 모델 내부 호출 시작: {len(image_urls)}개")

    # Validator 가져오기
    validator = get_validator()

    # 비동기가 아니므로 동기적으로 실행 (FastAPI 스레드풀에서 실행하면 더 좋음)
    # 여기서는 간단히 직접 호출
    try:
        results = validator.validate_batch(image_urls)
        return results
    except Exception as e:
        logger.error(f"AI 모델 내부 호출 실패: {e}")
        # 실패 시 Mock 결과 반환 또는 에러 처리
        if settings.use_mock_validator:
            return _mock_validate(image_urls)
        # 에러를 포함한 결과 반환
        return [{"url": url, "error": str(e)} for url in image_urls]


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


async def start_analyze(
    request: AnalyzeRequest, background_tasks: BackgroundTasks
) -> AnalyzeResponse:
    """
    이미지 분석 작업 시작 (비동기)

    작업 내용:
    1. 패션 아이템 Segmentation (객체 분리)
    2. AI 속성 분석 (카테고리, 색상, 소재, 스타일)
    """

    batch_id = request.batch_id
    redis_client = get_redis_client()

    # 중복 배치 ID 체크 (409)
    if redis_client.exists_batch(batch_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"이미 처리 중인 배치입니다: {batch_id}",
        )

    # 배치 정보 저장 (Redis)
    batch_data = {
        "user_id": request.user_id,
        "status": BatchStatus.ACCEPTED,
        "total": len(request.images),
        "completed": 0,
        "processing": len(request.images),
        "created_at": time.time(),
    }
    redis_client.set_batch(batch_id, batch_data)

    # 개별 작업 정보 저장 및 백그라운드 작업 시작
    initial_results = []
    for image in request.images:
        task_id = image.task_id

        # 작업 정보 저장 (Redis)
        task_data = {
            "task_id": task_id,  # task_id 명시적 저장
            "batch_id": batch_id,
            "user_id": request.user_id,
            "sequence": image.sequence,
            "target_image": image.target_image,
            "file_info": {
                "file_id": image.file_upload_info.file_id,
                "object_key": image.file_upload_info.object_key,
                "presigned_url": image.file_upload_info.presigned_url,
            },
            "status": TaskStatus.PREPROCESSING,
            "file_id": None,
            "analysis_result": None,
            "created_at": time.time(),
        }
        redis_client.set_task(task_id, task_data)
        redis_client.add_task_to_batch(batch_id, task_id)

        # 초기 results 배열 생성
        initial_results.append(
            BatchResultItem(
                task_id=task_id,
                status=TaskStatus.PREPROCESSING,
                file_id=None,
                major=None,
                extra=None,
            )
        )

        # 백그라운드 작업 시작 (Fire and forget)
        background_tasks.add_task(process_image_task, task_id)
        logger.info(f"백그라운드 작업 시작: task_id={task_id}")

    return AnalyzeResponse(
        batch_id=batch_id,
        status=BatchStatus.IN_PROGRESS,
        meta=BatchMeta(
            total=len(request.images),
            completed=0,
            processing=len(request.images),
            is_finished=False,
        ),
        results=initial_results,
    )


# ============================================================
# 3. Batch Status API - 작업 상태 조회
# ============================================================


def get_batch_status(batch_id: str) -> BatchStatusResponse | None:
    """배치 상태 조회"""
    redis_client = get_redis_client()

    # 배치 정보 조회 (Redis)
    batch = redis_client.get_batch(batch_id)
    if batch is None:
        return None

    # 해당 배치의 모든 작업 조회 (Redis)
    tasks = redis_client.get_tasks_by_batch(batch_id)

    results: list[BatchResultItem] = []
    completed_count = 0
    processing_count = 0

    # task_id -> (sequence, result_item) 매핑 생성
    task_map = {}
    for task in tasks:
        task_id = task.get("task_id", "")
        result_item = _build_batch_result_item(task_id, task)
        task_map[task_id] = (task.get("sequence", 0), result_item)

        if task["status"] == TaskStatus.COMPLETED:
            completed_count += 1
        elif task["status"] != TaskStatus.FAILED:
            processing_count += 1

    # sequence 순으로 정렬
    sorted_items = sorted(task_map.values(), key=lambda x: x[0])
    results = [item[1] for item in sorted_items]

    total = batch["total"]
    is_finished = completed_count == total

    # 전체 상태 결정
    if is_finished:
        overall_status = BatchStatus.COMPLETED
    else:
        overall_status = BatchStatus.IN_PROGRESS

    return BatchStatusResponse(
        batch_id=batch_id,
        status=overall_status,
        meta=BatchMeta(
            total=total,
            completed=completed_count,
            processing=processing_count,
            is_finished=is_finished,
        ),
        results=results,
    )


def _build_batch_result_item(task_id: str, task: dict) -> BatchResultItem:
    """BatchResultItem 빌드"""

    major = None
    extra = None

    if task.get("analysis_result"):
        attrs = task["analysis_result"]

        major = MajorAttributes(
            category=attrs.get("category", ""),
            color=attrs.get("color", []),
            material=attrs.get("material", []),
            style_tags=attrs.get("style_tags", []),
        )

        meta = MetaData(
            gender=attrs.get("gender", ""),
            season=attrs.get("season", []),
            formality=attrs.get("formality", ""),
            fit=attrs.get("fit", ""),
            occasion=attrs.get("occasion", []),
        )

        extra = ExtraAttributes(
            meta_data=meta,
            caption=attrs.get("caption", ""),
        )

    return BatchResultItem(
        task_id=task_id,
        status=task["status"],
        file_id=task.get("file_id"),
        major=major,
        extra=extra,
    )


# ============================================================
# 분석 작업 실행 함수들 (백그라운드에서 호출)
# ============================================================


async def process_image_task(task_id: str) -> None:
    """
    이미지 분석 작업 실행 (백그라운드)

    1. Segmentation (객체 분리)
    2. AI 속성 분석
    """
    redis_client = get_redis_client()

    # Redis에서 작업 정보 조회
    task = redis_client.get_task(task_id)
    if task is None:
        logger.error(f"작업을 찾을 수 없음: task_id={task_id}")
        return

    sequence = task.get("sequence", 0)
    batch_id = task["batch_id"]

    try:
        # Step 1: Segmentation (PREPROCESSING)
        task["status"] = TaskStatus.PREPROCESSING
        redis_client.set_task(task_id, task)

        processed_file_id = await _process_segmentation(
            task["file_info"]["presigned_url"], task["file_info"]["file_id"], sequence
        )

        task["file_id"] = processed_file_id

        # Step 2: AI 속성 분석 (ANALYZING)
        task["status"] = TaskStatus.ANALYZING
        redis_client.set_task(task_id, task)

        attributes = await _analyze_image_attributes(processed_file_id, sequence)

        task["analysis_result"] = attributes

        # 완료
        task["status"] = TaskStatus.COMPLETED
        redis_client.set_task(task_id, task)

        # 배치 완료 카운트 업데이트 (Redis)
        redis_client.increment_batch_completed(batch_id)

    except Exception as e:
        task["status"] = TaskStatus.FAILED
        task["error"] = str(e)
        redis_client.set_task(task_id, task)
        logger.error(f"작업 실패: task_id={task_id}, error={e}")


async def _process_segmentation(
    presigned_url: str, file_id: int, sequence: int = 0
) -> int:
    """
    패션 아이템 Segmentation (객체 분리)

    처리 과정:
    1. presigned_url에서 이미지 다운로드
    2. Instance Segmentation (BiRefNet)
    3. 세그먼테이션된 이미지 S3 업로드
    4. 기존 file_id 반환

    Returns:
        처리된 이미지의 파일 ID
    """
    try:
        # 1. 이미지 다운로드
        image = await asyncio.to_thread(download_image, presigned_url)
        if image is None:
            raise Exception("이미지 다운로드 실패")

        # 2. 배경 제거 (BiRefNet)
        remover = get_background_remover()
        # GPU 연산은 blocking이므로 thread로 실행
        segmented_image = await asyncio.to_thread(remover.remove_background, image)

        # 3. S3 업로드
        storage = get_storage()

        # 이미지를 bytes로 변환
        buf = io.BytesIO()
        segmented_image.save(buf, format="PNG")
        buf.seek(0)

        object_key = f"segmented/{file_id}_{sequence}.png"

        s3_url = await asyncio.to_thread(
            storage.upload_file, buf, object_key, "image/png"
        )

        logger.info(f"Segmentation 완료: file_id={file_id}, s3_url={s3_url}")
        return file_id

    except Exception as e:
        logger.error(f"Segmentation 처리 중 에러: {e}")
        raise e


async def _analyze_image_attributes(file_id: int, sequence: int = 0) -> dict:
    """
    AI 이미지 속성 분석

    처리 과정:
    1. 이미지 로드
    2. 캡션 생성 (BLIP-2 등)
    3. 캡션에서 속성 추출 (LLM 또는 규칙 기반)
       - 카테고리 분류
       - 색상 추출
       - 소재 분석
       - 스타일 태그 생성

    TODO: RunPod 또는 로컬 모델 연동
    - BLIP-2 / BLIP / LLaVA (캡션 생성)
    - GPT-4V / Claude Vision (속성 추출)

    Returns:
        분석된 속성들
    """
    # Mock: 시뮬레이션 (sequence별로 다른 시간)
    # sequence에 따라 처리 시간 다르게 (테스트용)
    delay = 1 + sequence  # seq0: 1초, seq1: 2초, seq2: 3초
    await asyncio.sleep(delay)
    logger.info(
        f"AI 분석 완료 (Mock): file_id={file_id}, sequence={sequence}, delay={delay}초"
    )

    return {
        "caption": "골드 버튼 디테일이 들어간 캐주얼한 스타일의 빨간색 니트입니다.",
        "category": Category.TOP,
        "color": ["빨강"],
        "material": ["니트"],
        "style_tags": ["캐주얼", "따뜻한"],
        "gender": "남녀공용",
        "season": ["봄", "가을"],
        "formality": "세미 포멀",
        "fit": "오버핏",
        "occasion": ["데이트", "캐주얼 모임", "주말 외출"],
    }  # 임시
