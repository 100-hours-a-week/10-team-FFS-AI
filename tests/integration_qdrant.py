import os
import sys

from fastapi.testclient import TestClient

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.closet.service import qdrant_service  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)

# 테스트 데이터
USER_ID = 999
CLOTHES_ID = 8888
TEST_IMAGE_URL = "https://example.com/test_image.jpg"
# 임의의 임베딩 벡터 (768 차원 - FashionSigLIP)
TEST_EMBEDDING = [0.1] * 768


def test_qdrant_integration_flow() -> None:
    print("\n[Test] Qdrant 통합 테스트 시작")

    # 1. 초기 상태: 중복 없어야 함
    # (직접 Qdrant 서비스 호출로 확인)
    results = qdrant_service.search_similar(
        TEST_EMBEDDING, user_id=USER_ID, score_threshold=0.99
    )
    assert len(results) == 0, "초기 상태에서 중복이 없어야 합니다."
    print("1. 초기 상태 확인 완료 (중복 없음)")

    # 2. 임베딩 저장 (POST /v1/closet/embedding)
    payload = {
        "userId": USER_ID,
        "clothesId": CLOTHES_ID,
        "embedding": TEST_EMBEDDING,
        "category": "TOP",
        "color": ["Black"],
        "material": ["Cotton"],
    }
    response = client.post("/v1/closet/embedding", json=payload)
    # Note: 스키마에 따라 Body는 리스트여야 함. Query param으로 user_id, clothes_id 전달

    assert response.status_code == 201
    print("2. 임베딩 저장 완료")

    # 3. 중복 검사 (이미 저장된 것과 동일한 벡터로 검색)
    # service._check_duplicate 함수를 직접 테스트하거나,
    # mock validator를 사용하여 validate api를 호출했을 때 중복이 뜨는지 확인
    # 여기서는 qdrant_service를 직접 사용하여 확인
    results = qdrant_service.search_similar(
        TEST_EMBEDDING, user_id=USER_ID, score_threshold=0.99
    )
    assert len(results) > 0, "저장 후에는 검색되어야 합니다."
    assert results[0].payload["user_id"] == USER_ID
    print(f"3. 중복 검사 확인 완료 (Score: {results[0].score})")

    # 4. 삭제 (DELETE /v1/closet/{clothesId})
    response = client.delete(f"/v1/closet/{CLOTHES_ID}")
    assert response.status_code == 200
    print("4. 아이템 삭제 완료")

    # 5. 삭제 확인
    results = qdrant_service.search_similar(
        TEST_EMBEDDING, user_id=USER_ID, score_threshold=0.99
    )
    assert len(results) == 0, "삭제 후에는 검색되지 않아야 합니다."
    print("5. 삭제 확인 완료")

    print("[Test] Qdrant 통합 테스트 성공!")


if __name__ == "__main__":
    test_qdrant_integration_flow()
