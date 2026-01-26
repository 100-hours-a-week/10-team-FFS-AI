from typing import Protocol

from app.embedding.schemas import ClothingMetadata


class EmbeddingTextFormatter(Protocol):
    def format(
        self: "EmbeddingTextFormatter", metadata: ClothingMetadata, caption: str
    ) -> str:
        ...


class HybridFormatter:
    """
    출력 예시:
        "검정, 네이비 울, 폴리에스터 코트. 포멀한 스타일. 겨울, 가을용.
         오버핏 더블 버튼 코트로 세련된 분위기를 연출합니다."
    """

    def format(
        self: "HybridFormatter", metadata: ClothingMetadata, caption: str
    ) -> str:
        parts: list[str] = []

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

        if metadata.formality:
            parts.append(f"{metadata.formality} 스타일")

        if metadata.season:
            season_str = ", ".join(metadata.season)
            parts.append(f"{season_str}용")

        if metadata.fit:
            parts.append(metadata.fit)

        if caption:
            parts.append(caption)

        return " ".join(parts)
