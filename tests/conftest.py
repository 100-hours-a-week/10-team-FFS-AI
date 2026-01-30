from __future__ import annotations

import os
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from app.embedding.service import EmbeddingService, get_embedding_service
from app.main import app


# ============================================================
# 🔧 환경변수 설정 (가장 먼저 실행)
# ============================================================
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> Generator[None, None, None]:
    """
    테스트 환경변수 설정
    - CI: GitHub Actions의 env 사용
    - 로컬: 테스트용 기본값 사용
    """
    # CI에서 설정된 환경변수가 있으면 그대로 사용, 없으면 로컬 테스트용 기본값
    test_env = {
        "APP_ENV": os.getenv("APP_ENV", "ci"),
        "DEBUG": os.getenv("DEBUG", "False"),
        # Qdrant 설정
        "QDRANT_HOST": os.getenv("QDRANT_HOST", "localhost"),
        "QDRANT_PORT": os.getenv("QDRANT_PORT", "6333"),
        "QDRANT_COLLECTION_NAME": os.getenv(
            "QDRANT_COLLECTION_NAME", "test_embeddings"
        ),
        # Redis 설정
        "REDIS_HOST": os.getenv("REDIS_HOST", "localhost"),
        "REDIS_PORT": os.getenv("REDIS_PORT", "6380"),
        "REDIS_DB": os.getenv("REDIS_DB", "0"),
        # API Keys (옵션)
        "UPSTAGE_API_KEY": os.getenv("UPSTAGE_API_KEY", "test_upstage_key"),
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", "test_access_key"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", "test_secret_key"),
    }

    # 환경변수 업데이트
    os.environ.update(test_env)

    # Settings 캐시 클리어 (중요!)
    from app.config import get_settings

    get_settings.cache_clear()

    # 디버깅 출력 (CI에서 확인용)
    print("\n" + "=" * 50)
    print("🔧 Test Environment Variables")
    print("=" * 50)
    for key, value in test_env.items():
        # API 키는 일부만 표시
        if "KEY" in key or "SECRET" in key:
            display_value = value[:10] + "..." if len(value) > 10 else value
        else:
            display_value = value
        print(f"{key}: {display_value}")
    print("=" * 50 + "\n")

    yield

    # 테스트 종료 후 캐시 정리
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def mock_db_clients(mocker: MockerFixture) -> AsyncMock:
    # Qdrant Mock
    mock_qdrant: AsyncMock = AsyncMock()
    mock_qdrant.get_collections.return_value = MagicMock(collections=[])
    mocker.patch("app.core.database.AsyncQdrantClient", return_value=mock_qdrant)

    # Redis Mock
    mock_redis: AsyncMock = AsyncMock()
    mocker.patch("app.core.database.Redis", return_value=mock_redis)
    mocker.patch("app.core.database.ConnectionPool", return_value=MagicMock())

    # health check mock
    mocker.patch(
        "app.core.database.check_health",
        new_callable=AsyncMock,
        return_value={"qdrant": "connected", "redis": "connected"},
    )

    return mock_qdrant


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_get_qdrant_client(
    mocker: MockerFixture, mock_db_clients: AsyncMock
) -> AsyncMock:
    mocker.patch(
        "app.embedding.service.get_qdrant_client", return_value=mock_db_clients
    )
    mocker.patch("app.core.database.get_qdrant_client", return_value=mock_db_clients)
    return mock_db_clients


@pytest.fixture
def mock_embedding_service() -> Generator[AsyncMock, None, None]:
    mock_service: AsyncMock = AsyncMock(spec=EmbeddingService)
    app.dependency_overrides[get_embedding_service] = lambda: mock_service
    yield mock_service
    app.dependency_overrides.clear()


@pytest.fixture
def mock_outfit_service() -> Generator[AsyncMock, None, None]:
    from app.outfit.service import OutfitService, get_outfit_service

    mock_service: AsyncMock = AsyncMock(spec=OutfitService)
    app.dependency_overrides[get_outfit_service] = lambda: mock_service
    yield mock_service
    app.dependency_overrides.clear()


# ============================================================
# Closet 테스트용 Fixtures
# ============================================================


@pytest.fixture(autouse=True)
def mock_redis_client(mocker: MockerFixture) -> MagicMock:
    """Redis 클라이언트 Mock (Closet 용)"""
    from app.core.redis_client import RedisClient

    # In-memory 저장소 (테스트용)
    batch_store = {}
    task_store = {}
    batch_task_sets = {}  # batch_id -> set of task_ids

    mock_redis = MagicMock(spec=RedisClient)

    def mock_set_batch(
        batch_id: str, data: dict[str, object], ttl: int = 86400
    ) -> None:
        batch_store[batch_id] = data

    def mock_get_batch(batch_id: str) -> dict[str, object] | None:
        return batch_store.get(batch_id)

    def mock_exists_batch(batch_id: str) -> bool:
        return batch_id in batch_store

    def mock_set_task(task_id: str, data: dict[str, object], ttl: int = 86400) -> None:
        task_store[task_id] = data

    def mock_get_task(task_id: str) -> dict[str, object] | None:
        return task_store.get(task_id)

    def mock_add_task_to_batch(batch_id: str, task_id: str, ttl: int = 86400) -> None:
        if batch_id not in batch_task_sets:
            batch_task_sets[batch_id] = set()
        batch_task_sets[batch_id].add(task_id)

    def mock_get_tasks_by_batch(batch_id: str) -> list[dict[str, object]]:
        task_ids = batch_task_sets.get(batch_id, set())
        return [task_store[tid] for tid in task_ids if tid in task_store]

    def mock_increment_batch_completed(batch_id: str) -> None:
        if batch_id in batch_store:
            batch_store[batch_id]["completed"] = (
                batch_store[batch_id].get("completed", 0) + 1
            )
            batch_store[batch_id]["processing"] = max(
                0, batch_store[batch_id].get("processing", 0) - 1
            )

    # 메서드 연결
    mock_redis.set_batch.side_effect = mock_set_batch
    mock_redis.get_batch.side_effect = mock_get_batch
    mock_redis.exists_batch.side_effect = mock_exists_batch
    mock_redis.set_task.side_effect = mock_set_task
    mock_redis.get_task.side_effect = mock_get_task
    mock_redis.add_task_to_batch.side_effect = mock_add_task_to_batch
    mock_redis.get_tasks_by_batch.side_effect = mock_get_tasks_by_batch
    mock_redis.increment_batch_completed.side_effect = mock_increment_batch_completed
    mock_redis.ping.return_value = True

    # get_redis_client가 mock을 반환하도록 패치
    mocker.patch(
        "app.closet.service.get_redis_client",
        return_value=mock_redis,
    )
    mocker.patch(
        "app.core.redis_client.get_redis_client",
        return_value=mock_redis,
    )

    return mock_redis


@pytest.fixture(autouse=True)
def mock_download_image(mocker: MockerFixture) -> MagicMock:
    """이미지 다운로드 함수 Mock"""
    from PIL import Image

    # 가짜 이미지 생성 함수
    def fake_download_image(url: str, timeout: float = 30.0) -> Image.Image | None:
        # URL에 따라 다른 이미지 반환 (테스트용)
        if "error" in url.lower():
            return None

        # 100x100 빨간색 이미지 반환
        return Image.new("RGB", (100, 100), color="red")

    # download_image 함수를 mock으로 교체
    mock = mocker.patch(
        "app.closet.validators.download_image",
        side_effect=fake_download_image,
    )
    mocker.patch(
        "app.closet.service.download_image",
        side_effect=fake_download_image,
    )

    return mock


@pytest.fixture(autouse=True)
def mock_httpx_client(mocker: MockerFixture) -> MagicMock:
    """HTTPX 클라이언트 Mock (파일 크기 체크용)"""
    # HEAD 요청 응답 Mock (파일 크기 조회용)
    mock_head_response = MagicMock()
    mock_head_response.headers = {"content-length": "5000000"}  # 5MB (10MB 이하)

    # HTTPX Client Mock
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.head = MagicMock(return_value=mock_head_response)

    # httpx.Client를 mock으로 교체
    mocker.patch("httpx.Client", return_value=mock_client)

    return mock_client


@pytest.fixture(autouse=True)
def mock_s3_storage(mocker: MockerFixture) -> MagicMock:
    """S3 Storage Mock"""
    from app.core.storage import S3Storage

    mock_storage = MagicMock(spec=S3Storage)

    def mock_upload_file(
        file_obj: MagicMock, object_key: str, content_type: str = "image/png"
    ) -> str:
        # 가짜 S3 URL 반환
        return f"https://s3.example.com/{object_key}"

    mock_storage.upload_file.side_effect = mock_upload_file

    # get_storage가 mock을 반환하도록 패치
    mocker.patch(
        "app.closet.service.get_storage",
        return_value=mock_storage,
    )
    mocker.patch(
        "app.core.storage.get_storage",
        return_value=mock_storage,
    )

    return mock_storage


@pytest.fixture(autouse=True)
def mock_background_remover(mocker: MockerFixture) -> MagicMock:
    """BiRefNet 배경 제거 Mock"""
    from PIL import Image

    mock_remover = MagicMock()

    def mock_remove_background(image: Image.Image) -> Image.Image:
        # 입력 이미지에 alpha channel 추가해서 반환
        rgba_image = image.convert("RGBA")
        return rgba_image

    mock_remover.remove_background.side_effect = mock_remove_background

    # get_background_remover가 mock을 반환하도록 패치
    mocker.patch(
        "app.closet.service.get_background_remover",
        return_value=mock_remover,
    )
    mocker.patch(
        "app.closet.background_removal.get_background_remover",
        return_value=mock_remover,
    )

    return mock_remover
