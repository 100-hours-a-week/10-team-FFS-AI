import asyncio
import io
import logging

from langfuse import observe
from PIL import Image

from app.closet.s3_client import S3Client
from app.outfit.schemas import Outfit, OutfitResponse, UploadSlot
from app.outfit.vton_client import VTONClient, VTONRequest

logger = logging.getLogger(__name__)


class VTONProcessor:
    def __init__(self, vton_client: VTONClient | None = None) -> None:
        self.vton_client = vton_client or VTONClient()
        self.s3_client = S3Client()

    @observe(name="vton_processor.process")
    async def process(
        self,
        response: OutfitResponse,
        upload_slots: list[UploadSlot],
    ) -> None:
        """각 코디에 대해 VTON 이미지 생성 및 S3 업로드 (병렬 처리)"""
        # 병렬 처리할 태스크 목록 생성
        tasks = []
        for i, outfit in enumerate(response.outfits):
            if i >= len(upload_slots):
                logger.warning(f"Not enough upload slots for outfit {i}")
                break

            slot = upload_slots[i]
            # 개별 코디 처리를 태스크로 추가 (await 하지 않음)
            tasks.append(self._process_single_outfit(outfit, slot))

        if tasks:
            # 모든 태스크를 동시에 실행 (병렬 처리)
            logger.info(f"Starting parallel VTON processing for {len(tasks)} outfits")
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("Parallel VTON processing completed")

    async def _process_single_outfit(
        self,
        outfit: Outfit,
        slot: UploadSlot,
    ) -> None:
        """개별 코디에 대한 VTON 이미지 생성 및 S3 업로드"""
        garment_urls = [item.image_url for item in outfit.items]

        if not garment_urls:
            outfit.vton_error = "의류 이미지 URL을 찾을 수 없음"
            return

        try:
            vton_request = VTONRequest(image_urls=garment_urls)
            vton_response = await self.vton_client.generate_outfit_image(vton_request)

            if vton_response.status == "completed" and vton_response.image_data:
                # [Standardization] WEBP/PNG -> JPEG 변환
                with Image.open(io.BytesIO(vton_response.image_data)) as img:
                    img = img.convert("RGB")  # 투명도 제거 및 호환성 확보
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=95)
                    final_image_data = buffer.getvalue()

                logger.info(
                    f"Converted image format to JPEG for outfit {outfit.outfit_id} "
                    f"(Size: {len(final_image_data)} bytes)"
                )

                await self.s3_client.put_image(
                    slot.presigned_url, final_image_data, content_type="image/jpeg"
                )
                outfit.file_id = slot.file_id
                logger.info(
                    f"VTON completed for outfit {outfit.outfit_id}, file_id={slot.file_id}"
                )
            else:
                outfit.vton_error = vton_response.error or "이미지 생성 실패"
                logger.warning(
                    f"VTON generation failed for outfit {outfit.outfit_id}: "
                    f"{vton_response.error}"
                )

        except Exception as e:
            outfit.vton_error = f"VTON 처리 실패: {e}"
            logger.exception(
                f"Exception during VTON processing for outfit {outfit.outfit_id}"
            )
