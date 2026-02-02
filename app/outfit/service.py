import logging
from functools import lru_cache

from app.outfit.llm_client import OpenAIClient
from app.outfit.outfit_composer import OutfitComposer
from app.outfit.query_parser import QueryParser
from app.outfit.repository import ClothingRepository
from app.outfit.schemas import OutfitRequest, OutfitResponse, UploadSlot
from app.outfit.search_query_builder import SearchQueryBuilder
from app.outfit.vton_client import VTONClient, VTONRequest, upload_to_s3

logger = logging.getLogger(__name__)


class OutfitService:
    def __init__(
        self,
        query_parser: QueryParser | None = None,
        search_builder: SearchQueryBuilder | None = None,
        repository: ClothingRepository | None = None,
        composer: OutfitComposer | None = None,
        vton_client: VTONClient | None = None,
    ) -> None:
        self.query_parser = query_parser or QueryParser(llm_client=OpenAIClient())
        self.search_builder = search_builder or SearchQueryBuilder()
        self.repository = repository or ClothingRepository()
        self.composer = composer or OutfitComposer()
        self.vton_client = vton_client or VTONClient()

    async def recommend(self, request: OutfitRequest) -> OutfitResponse:
        logger.info(f"Processing outfit request for user: {request.user_id}")

        parsed = await self.query_parser.parse(request.query)
        logger.info(f"Parsed query: occasion={parsed.occasion}, style={parsed.style}")

        search_queries = self.search_builder.build(parsed)
        logger.info(f"Generated {len(search_queries)} search queries")

        search_results = await self.repository.search_multiple(
            user_id=request.user_id,
            queries=search_queries,
        )

        total_candidates = sum(len(r.candidates) for r in search_results)
        logger.info(f"Found {total_candidates} total candidates")

        response = await self.composer.compose(
            parsed_query=parsed,
            search_results=search_results,
        )
        response.session_id = request.session_id

        logger.info(f"Generated {len(response.outfits)} outfit recommendations")

        # VTON 이미지 생성 (urls가 있는 경우에만)
        if request.urls:
            await self._generate_vton_images(response, request.urls)
        else:
            # urls가 없으면 VTON 미요청으로 표시
            for outfit in response.outfits:
                outfit.vton_error = "VTON 미요청 (urls 없음)"

        return response

    async def _generate_vton_images(
        self, response: OutfitResponse, upload_slots: list[UploadSlot]
    ) -> None:
        """각 코디에 대해 VTON 이미지 생성 및 S3 업로드"""
        for i, outfit in enumerate(response.outfits):
            if i >= len(upload_slots):
                logger.warning(f"Not enough upload slots for outfit {i}")
                break

            slot = upload_slots[i]
            garment_urls = [item.image_url for item in outfit.items]

            # VTON 이미지 생성
            vton_request = VTONRequest(image_urls=garment_urls)
            vton_response = await self.vton_client.generate_outfit_image(vton_request)

            if vton_response.status == "completed" and vton_response.image_data:
                # S3 업로드
                success = await upload_to_s3(
                    slot.presigned_url, vton_response.image_data
                )
                if success:
                    outfit.file_id = slot.file_id
                    logger.info(
                        f"VTON completed for outfit {outfit.outfit_id}, file_id={slot.file_id}"
                    )
                else:
                    outfit.vton_error = "S3 업로드 실패"
                    logger.warning(f"S3 upload failed for outfit {outfit.outfit_id}")
            else:
                outfit.vton_error = vton_response.error or "이미지 생성 실패"
                logger.warning(
                    f"VTON generation failed for outfit {outfit.outfit_id}: {vton_response.error}"
                )


# FastAPI 의존성 주입용 (DI 컨테이너 도입 전까지 사용)
@lru_cache
def get_outfit_service() -> OutfitService:
    return OutfitService()
