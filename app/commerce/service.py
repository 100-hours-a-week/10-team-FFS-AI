"""커머스 배치 서비스.

키워드 단위로 크롤링 → 중복 체크 → 이미지 분석(vLLM) → 임베딩 → Qdrant 저장.
배치 시간 윈도우(KST 2~6시)를 초과하면 진행 상태를 Redis에 저장하고 중단한다.
다음 날 배치 시 저장된 위치부터 이어서 처리한다.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

import httpx

from app.closet.model_analyzer import ModelServerAnalyzer
from app.closet.schemas import ExtraAttributes, ExtraMetadata, MajorAttributes
from app.commerce.crawler import SEARCH_KEYWORDS, NaverShoppingCrawler
from app.commerce.repository import CommerceRepository
from app.commerce.schemas import BatchResult, NaverProduct
from app.config import Settings, get_settings
from app.core.database import get_redis_client
from app.embedding.client import UpstageEmbeddingClient
from app.embedding.formatter import HybridFormatter

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

# Redis 키: 배치 진행 상태 저장
REDIS_PROGRESS_KEY = "commerce:batch:keyword_index"

# 키워드 리스트를 (카테고리, 키워드) 튜플로 평탄화
KEYWORD_LIST: list[tuple[str, str]] = []
for _cat, _kws in SEARCH_KEYWORDS.items():
    for _kw in _kws:
        KEYWORD_LIST.append((_cat, _kw))


class CommerceBatchService:
    """커머스 상품 배치 처리 서비스."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.crawler = NaverShoppingCrawler(self.settings)
        self.repository = CommerceRepository(settings=self.settings)
        self.formatter = HybridFormatter()
        self.embedding_client = UpstageEmbeddingClient()

    async def is_vllm_available(self) -> bool:
        """vLLM 서버 헬스체크."""
        vllm_url = self.settings.vllm_server_url
        if not vllm_url:
            logger.warning("VLLM_SERVER_URL이 설정되지 않음")
            return False

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{vllm_url.rstrip('/')}/health")
                return resp.status_code == 200
        except Exception as e:
            logger.warning("vLLM 서버 헬스체크 실패: %s", e)
            return False

    # ── 진행 상태 관리 (Redis) ──

    async def _get_progress(self) -> int:
        """Redis에서 마지막 처리된 키워드 인덱스를 가져온다."""
        try:
            redis = get_redis_client()
            val = await redis.get(REDIS_PROGRESS_KEY)
            return int(val) if val else 0
        except Exception:
            return 0

    async def _save_progress(self, index: int) -> None:
        """현재 키워드 인덱스를 Redis에 저장한다."""
        try:
            redis = get_redis_client()
            await redis.set(REDIS_PROGRESS_KEY, str(index))
        except Exception as e:
            logger.warning("진행 상태 저장 실패: %s", e)

    async def _reset_progress(self) -> None:
        """전체 키워드 1회전 완료 시 진행 상태를 초기화한다."""
        try:
            redis = get_redis_client()
            await redis.delete(REDIS_PROGRESS_KEY)
        except Exception as e:
            logger.warning("진행 상태 초기화 실패: %s", e)

    # ── 시간 윈도우 체크 ──

    def _is_within_window(self) -> bool:
        """현재 시각이 배치 시간 윈도우(KST) 내인지 확인한다."""
        now = datetime.now(KST)
        return now.hour < self.settings.batch_end_hour

    # ── 배치 실행 ──

    async def run_batch(self) -> BatchResult:
        """배치 파이프라인을 키워드 단위로 실행한다.

        - Redis에서 마지막 진행 위치를 불러와 이어서 처리
        - 배치 시간 윈도우(KST 2~6시)를 초과하면 중단하고 진행 상태 저장
        - 전체 키워드 1회전 완료 시 진행 상태 초기화 (다음 날 처음부터)
        """
        start_time = time.perf_counter()
        result = BatchResult()

        start_index = await self._get_progress()
        total_keywords = len(KEYWORD_LIST)

        if start_index >= total_keywords:
            # 이전 회차가 완료된 상태 → 처음부터 시작
            start_index = 0
            await self._reset_progress()

        logger.info(
            "===== 커머스 배치 시작 (키워드 %d/%d부터) =====",
            start_index + 1,
            total_keywords,
        )

        analyzer = ModelServerAnalyzer()
        seen_ids: set[str] = set()
        stopped_early = False

        async with httpx.AsyncClient(timeout=10.0) as http_client:
            for idx in range(start_index, total_keywords):
                # 시간 윈도우 체크
                if not self._is_within_window():
                    logger.info(
                        "배치 시간 윈도우 초과 (KST %d시) → 중단 (키워드 %d/%d)",
                        self.settings.batch_end_hour,
                        idx,
                        total_keywords,
                    )
                    await self._save_progress(idx)
                    stopped_early = True
                    break

                category, keyword = KEYWORD_LIST[idx]
                logger.info(
                    "[%d/%d] 크롤링: '%s' (카테고리: %s)",
                    idx + 1,
                    total_keywords,
                    keyword,
                    category,
                )

                # 키워드 크롤링
                try:
                    products = await self.crawler.crawl_keyword(
                        http_client,
                        keyword,
                        self.settings.batch_max_products_per_keyword,
                    )
                except Exception as e:
                    logger.error("키워드 '%s' 크롤링 실패: %s", keyword, e)
                    continue

                # 크롤링 중복 제거
                unique_products: list[NaverProduct] = []
                for p in products:
                    if p.product_id not in seen_ids:
                        seen_ids.add(p.product_id)
                        unique_products.append(p)

                result.total_fetched += len(unique_products)

                # 상품별 처리
                for product in unique_products:
                    try:
                        status = await self._process_product(
                            product,
                            category,
                            analyzer,
                        )
                        if status == "skipped":
                            result.total_skipped += 1
                        elif status == "updated":
                            result.total_upserted += 1
                        elif status == "new":
                            result.total_analyzed += 1
                            result.total_upserted += 1
                    except Exception as e:
                        result.total_failed += 1
                        logger.error(
                            "상품 처리 실패: product_id=%s, error=%s",
                            product.product_id,
                            e,
                        )
                    await asyncio.sleep(0.5)

                # 키워드 처리 완료 → 진행 상태 저장
                await self._save_progress(idx + 1)
                logger.info(
                    "  -> 키워드 '%s' 완료: %d개 처리",
                    keyword,
                    len(unique_products),
                )

        # 전체 키워드 1회전 완료
        if not stopped_early:
            await self._reset_progress()
            logger.info("전체 키워드 1회전 완료 → 진행 상태 초기화")

        result.duration_sec = time.perf_counter() - start_time

        logger.info(
            "===== 커머스 배치 %s =====\n"
            "  크롤링: %d개\n"
            "  스킵(중복): %d개\n"
            "  분석(신규): %d개\n"
            "  저장 성공: %d개\n"
            "  실패: %d개\n"
            "  소요 시간: %.0f초 (%.1f분)",
            "중단" if stopped_early else "완료",
            result.total_fetched,
            result.total_skipped,
            result.total_analyzed,
            result.total_upserted,
            result.total_failed,
            result.duration_sec,
            result.duration_sec / 60,
        )

        return result

    async def _process_product(
        self,
        product: NaverProduct,
        category: str,
        analyzer: ModelServerAnalyzer,
    ) -> str:
        """상품 1개를 처리한다.

        Returns:
            "skipped" - 이미 존재하고 변동 없음
            "updated" - 이미 존재하지만 가격/제목 변경 → payload만 업데이트
            "new"     - 신규 상품 → 이미지 분석 + 임베딩 + Qdrant 저장
        """
        existing = await self.repository.get_existing_product(product.product_id)

        if existing is not None:
            if not self.repository.is_product_changed(
                existing, product.price, product.title
            ):
                return "skipped"

            await self.repository.update_payload(
                product.product_id,
                {"price": product.price, "title": product.title},
            )
            return "updated"

        return await self._analyze_and_store(product, category, analyzer)

    async def _analyze_and_store(
        self,
        product: NaverProduct,
        category: str,
        analyzer: ModelServerAnalyzer,
    ) -> str:
        """신규 상품: 이미지 다운로드 → vLLM 분석 → 임베딩 → Qdrant 저장."""
        # 1. 이미지 다운로드
        image_bytes = await self._download_image(product.image_url)

        # 2. vLLM 이미지 분석
        analysis = await analyzer.analyze_image(image_bytes)

        major = MajorAttributes(
            category=analysis.major.category,
            color=analysis.major.color,
            material=analysis.major.material,
            style_tags=analysis.major.style_tags,
        )
        extra = ExtraAttributes(
            meta_data=ExtraMetadata(
                gender=analysis.extra.meta_data.gender,
                season=analysis.extra.meta_data.season,
                formality=analysis.extra.meta_data.formality,
                fit=analysis.extra.meta_data.fit,
                occasion=analysis.extra.meta_data.occasion,
            ),
            caption=analysis.extra.caption,
        )

        # 3. 임베딩 텍스트 생성 + 벡터 생성
        embedding_text = self.formatter.format(major, extra)
        vector = await self.embedding_client.embed(embedding_text)

        # 4. Qdrant payload 구성 + 저장
        payload = self._build_payload(product, major, extra, embedding_text)
        success = await self.repository.upsert(
            product.product_id,
            vector,
            payload,
        )

        if not success:
            raise RuntimeError(f"Qdrant upsert 실패: {product.product_id}")

        return "new"

    @staticmethod
    async def _download_image(url: str) -> bytes:
        """이미지 URL에서 바이트를 다운로드한다."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    @staticmethod
    def _build_payload(
        product: NaverProduct,
        major: MajorAttributes,
        extra: ExtraAttributes,
        embedding_text: str,
    ) -> dict:
        """Qdrant에 저장할 payload를 구성한다.

        기존 shop/repository.py의 _to_candidate()와 호환되는 구조.
        """
        return {
            "productId": product.product_id,
            "title": product.title,
            "link": product.link,
            "imageUrl": product.image_url,
            "price": product.price,
            "brand": product.brand or "",
            "source": "naver",
            "category": major.category,
            "color": major.color,
            "material": major.material,
            "styleTags": major.style_tags,
            "season": extra.meta_data.season,
            "formality": extra.meta_data.formality or "",
            "occasion": extra.meta_data.occasion,
            "embeddingText": embedding_text,
        }
