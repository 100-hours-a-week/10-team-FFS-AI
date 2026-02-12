from typing import Any, Protocol


class ImageAnalyzer(Protocol):
    async def analyze_image(self, image_bytes: bytes) -> dict[str, Any]: ...
