import logging

from app.closet.s3_client import S3Client
from app.outfit.schemas import OutfitResponse, UploadSlot
from app.outfit.vton_client import VTONClient, VTONRequest

logger = logging.getLogger(__name__)


class VTONProcessor:
    def __init__(self, vton_client: VTONClient | None = None) -> None:
        self.vton_client = vton_client or VTONClient()
        self.s3_client = S3Client()

    async def process(
        self,
        response: OutfitResponse,
        upload_slots: list[UploadSlot],
    ) -> None:
        """각 코디에 대해 VTON 이미지 생성 및 S3 업로드"""
        for i, outfit in enumerate(response.outfits):
            if i >= len(upload_slots):
                logger.warning(f"Not enough upload slots for outfit {i}")
                # 슬롯이 모자라도 남은 의류는 처리하거나, 여기서 멈추거나 정책 결정 필요
                # 일단은 break
                break

            slot = upload_slots[i]
            garment_urls = [item.image_url for item in outfit.items]

            if not garment_urls:
                outfit.vton_error = "의류 이미지 URL을 찾을 수 없음"
                continue

            vton_request = VTONRequest(image_urls=garment_urls)
            vton_response = await self.vton_client.generate_outfit_image(vton_request)

            if vton_response.status == "completed" and vton_response.image_data:
                try:
                    # [Standardization] WEBP/PNG -> JPEG 변환
                    import io

                    from PIL import Image

                    with Image.open(io.BytesIO(vton_response.image_data)) as img:
                        img = img.convert("RGB")  # 투명도 제거 및 호환성 확보
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG", quality=95)
                        final_image_data = buffer.getvalue()

                    logger.info(
                        f"Converted image format to JPEG (Size: {len(final_image_data)} bytes)"
                    )

                    await self.s3_client.put_image(
                        slot.presigned_url, final_image_data, content_type="image/jpeg"
                    )
                    outfit.file_id = slot.file_id
                    logger.info(
                        f"VTON completed for outfit {outfit.outfit_id}, file_id={slot.file_id}"
                    )
                except Exception as e:
                    outfit.vton_error = "S3 업로드 실패"
                    logger.warning(
                        f"S3 upload failed for outfit {outfit.outfit_id}: {e}"
                    )
            else:
                outfit.vton_error = vton_response.error or "이미지 생성 실패"
                logger.warning(
                    f"VTON generation failed for outfit {outfit.outfit_id}: {vton_response.error}"
                )
