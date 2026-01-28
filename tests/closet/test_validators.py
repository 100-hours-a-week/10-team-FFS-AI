"""
Validator 단위 테스트 - AI 모델 동작 검증

네트워크 없이 Mock Validator 로직만 테스트합니다.
"""

import pytest

from app.closet.validators import MockImageValidator


def test_mock_validator_nsfw_detection() -> None:
    """Mock Validator NSFW 탐지 테스트"""
    validator = MockImageValidator()

    # Given: NSFW 키워드가 포함된 URL
    nsfw_url = "https://example.com/nsfw_image.jpg"

    # When: 검증 실행
    result = validator.validate_image(nsfw_url)

    # Then: NSFW로 탐지되어야 함
    assert result["url"] == nsfw_url
    assert result["nsfw"]["is_nsfw"] is True
    assert result["nsfw"]["score"] >= 0.5
    assert result["error"] is None


def test_mock_validator_normal_image() -> None:
    """Mock Validator 정상 이미지 테스트"""
    validator = MockImageValidator()

    # Given: 정상 패션 이미지 URL
    normal_url = "https://example.com/shirt.jpg"

    # When: 검증 실행
    result = validator.validate_image(normal_url)

    # Then: 모든 검증 통과
    assert result["url"] == normal_url
    assert result["nsfw"]["is_nsfw"] is False
    assert result["fashion"]["is_fashion"] is True
    assert result["error"] is None


def test_mock_validator_not_fashion() -> None:
    """Mock Validator 패션 아님 탐지 테스트"""
    validator = MockImageValidator()

    # Given: 음식 이미지 URL
    food_url = "https://example.com/food.jpg"

    # When: 검증 실행
    result = validator.validate_image(food_url)

    # Then: 패션 아님으로 탐지
    assert result["url"] == food_url
    assert result["nsfw"]["is_nsfw"] is False
    assert result["fashion"]["is_fashion"] is False
    assert result["error"] is None


def test_mock_validator_batch() -> None:
    """Mock Validator 배치 처리 테스트"""
    validator = MockImageValidator()

    # Given: 여러 이미지 URL
    urls = [
        "https://example.com/shirt.jpg",  # 정상
        "https://example.com/nsfw_image.jpg",  # NSFW
        "https://example.com/food.jpg",  # 패션 아님
    ]

    # When: 배치 검증 실행
    results = validator.validate_batch(urls)

    # Then: 각각 올바르게 분류되어야 함
    assert len(results) == 3

    # 첫 번째: 정상
    assert results[0]["nsfw"]["is_nsfw"] is False
    assert results[0]["fashion"]["is_fashion"] is True

    # 두 번째: NSFW
    assert results[1]["nsfw"]["is_nsfw"] is True

    # 세 번째: 패션 아님
    assert results[2]["fashion"]["is_fashion"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
