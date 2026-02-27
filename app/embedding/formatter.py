from typing import Protocol

from app.closet.schemas import ExtraAttributes, MajorAttributes


class EmbeddingTextFormatter(Protocol):
    def format(
        self: "EmbeddingTextFormatter", major: MajorAttributes, extra: ExtraAttributes
    ) -> str: ...


class HybridFormatter:
    def format(
        self: "HybridFormatter", major: MajorAttributes, extra: ExtraAttributes
    ) -> str:
        parts: list[str] = []
        meta = extra.meta_data

        color_str = ", ".join(major.color) if major.color else ""
        material_str = ", ".join(major.material) if major.material else ""

        if color_str and material_str:
            parts.append(f"{color_str} {material_str} {major.category}")
        elif color_str:
            parts.append(f"{color_str} {major.category}")
        elif material_str:
            parts.append(f"{material_str} {major.category}")
        else:
            parts.append(major.category)

        if meta.formality:
            parts.append(f"{meta.formality} 스타일")

        if meta.season:
            season_str = ", ".join(meta.season)
            parts.append(f"{season_str}용")

        if meta.fit:
            parts.append(meta.fit)

        if major.style_tags:
            style_tags_str = ", ".join(major.style_tags)
            parts.append(style_tags_str)

        if meta.occasion:
            occasion_str = ", ".join(meta.occasion)
            parts.append(f"{occasion_str}에 적합")

        if extra.caption:
            parts.append(extra.caption)

        return ". ".join(parts)
