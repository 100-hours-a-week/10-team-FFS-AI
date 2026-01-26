from typing import Protocol

from app.embedding.schemas import ClothingMetadata


class EmbeddingTextFormatter(Protocol):
    """임베딩 텍스트 포맷터 인터페이스

    메타데이터와 캡션을 조합하여 임베딩 모델 입력 텍스트를 생성합니다.
    """

    def format(self, metadata: ClothingMetadata, caption: str) -> str:
        """임베딩용 텍스트 생성

        Args:
            metadata: 의류 메타데이터
            caption: AI가 생성한 이미지 캡션

        Returns:
            임베딩 모델에 입력할 텍스트
        """
        ...


class HybridFormatter:
    """하이브리드 포맷터

    구조화된 메타데이터와 자연어 캡션을 조합합니다.

    출력 예시:
        "검정, 네이비 울, 폴리에스터 코트. 포멀한 스타일. 겨울, 가을용.
         오버핏 더블 버튼 코트로 세련된 분위기를 연출합니다."
    """

    def format(self, metadata: ClothingMetadata, caption: str) -> str:
        parts = []

        # 색상 + 소재 + 카테고리
        color_str = ", ".join(metadata.color) if metadata.color else ""
        material_str = ", ".join(metadata.material) if metadata.material else ""

        if color_str and material_str:
            parts.append(f"{color_str} {material_str} {metadata.category}")
        elif color_str:
            parts.append(f"{color_str} {metadata.category}")
        elif material_str:
            parts.append(f"{material_str} {metadata.category}")
        else:
            parts.append(metadata.category)

        # 스타일
        if metadata.formality:
            parts.append(f"{metadata.formality} 스타일")

        # 계절
        if metadata.season:
            season_str = ", ".join(metadata.season)
            parts.append(f"{season_str}용")

        # 핏
        if metadata.fit:
            parts.append(f"{metadata.fit}")

        # 캡션
        if caption:
            parts.append(caption)

        return ". ".join(parts)
