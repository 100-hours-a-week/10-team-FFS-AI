#!/usr/bin/env python3
"""
커머스 상품 DB 크롤러
====================
네이버 쇼핑 API → Gemini 이미지 분석 → Upstage 임베딩 → Qdrant commerce 컬렉션

실행: python scripts/commerce_crawler.py
"""

import asyncio
import logging
import re
import time

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

from app.closet.gemini_client import GeminiImageAnalyzer
from app.closet.schemas import ExtraAttributes, ExtraMetadata, MajorAttributes
from app.config import get_settings
from app.core.database import close_databases, get_qdrant_client, init_databases
from app.embedding.client import UpstageEmbeddingClient
from app.embedding.formatter import HybridFormatter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


SEARCH_KEYWORDS = [
    # 상의
    "남성 반팔 티셔츠",
    "남성 맨투맨",
    "남성 후드",
    "여성 블라우스",
    "여성 니트",
    # 하의
    "남성 슬랙스",
    "남성 청바지",
    "여성 와이드팬츠",
    "여성 스커트",
    # 원피스
    "미디 원피스",
    "롱 원피스",
    "미니 원피스",
    "셔츠 원피스",
    # 신발
    "스니커즈",
    "로퍼",
    "여성 부츠",
    # 가방
    "크로스백",
    "토트백",
    "숄더백",
    "백팩",
    # 악세서리
    "남성 모자",
    "스카프",
    "귀걸이",
    "목걸이",
]

ITEMS_PER_KEYWORD = 10
TARGET_TOTAL = 240

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유틸리티 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def strip_html(text: str) -> str:
    """HTML 태그 제거 (네이버 API 응답의 <b> 태그 등)"""
    return re.sub(r"<[^>]+>", "", text)


def clean_naver_item(item: dict, keyword: str) -> dict:
    """네이버 API 응답 1개 → 정제된 상품 dict"""
    return {
        "productId": item.get("productId", ""),
        "title": strip_html(item.get("title", "")),
        "link": item.get("link", ""),
        "imageUrl": item.get("image", ""),
        "price": int(item.get("lprice", "0")),
        "mallName": item.get("mallName", ""),
        "brand": item.get("brand", "") or None,
        "maker": item.get("maker", "") or None,
        "category1": item.get("category1", ""),
        "category2": item.get("category2", "") or None,
        "category3": item.get("category3", "") or None,
        "category4": item.get("category4", "") or None,
        "searchKeyword": keyword,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: 네이버 쇼핑 API 크롤링
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def search_naver_shopping(
    client: httpx.AsyncClient,
    query: str,
    display: int = 10,
    start: int = 1,
) -> list[dict]:
    """네이버 쇼핑 검색 API 호출 → 정제된 상품 리스트 반환"""
    settings = get_settings()

    response = await client.get(
        "https://openapi.naver.com/v1/search/shop.json",
        headers={
            "X-Naver-Client-Id": settings.naver_client_id,
            "X-Naver-Client-Secret": settings.naver_client_secret,
        },
        params={
            "query": query,
            "display": display,
            "start": start,
            "sort": "sim",
        },
    )
    response.raise_for_status()

    return [clean_naver_item(item, query) for item in response.json().get("items", [])]


async def crawl_all_products() -> list[dict]:
    """전체 키워드 순회 → 중복 제거 → 최대 TARGET_TOTAL개 반환"""
    all_products: dict[str, dict] = {}

    async with httpx.AsyncClient(timeout=10.0) as client:
        for i, keyword in enumerate(SEARCH_KEYWORDS):
            logger.info(f"[{i + 1}/{len(SEARCH_KEYWORDS)}] 검색: '{keyword}'")

            try:
                items = await search_naver_shopping(
                    client, keyword, display=ITEMS_PER_KEYWORD
                )
                new_count = sum(
                    1 for item in items if item["productId"] not in all_products
                )
                for item in items:
                    all_products.setdefault(item["productId"], item)

                logger.info(
                    f"  → {len(items)}개 수집, {new_count}개 신규 "
                    f"(누적: {len(all_products)}개)"
                )

                if len(all_products) >= TARGET_TOTAL:
                    logger.info(f"목표 {TARGET_TOTAL}개 달성!")
                    break

            except Exception as e:
                logger.error(f"  → 검색 실패: {e}")

            await asyncio.sleep(0.15)  # Rate limit 방지

    products = list(all_products.values())[:TARGET_TOTAL]
    logger.info(f"총 {len(products)}개 상품 수집 완료")
    return products


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: 이미지 분석 + 임베딩 + Qdrant 저장
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def download_image(url: str) -> bytes:
    """이미지 URL에서 바이트 다운로드"""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


def parse_analysis(analysis: dict) -> tuple[MajorAttributes, ExtraAttributes]:
    """Gemini 분석 결과 dict → Pydantic 모델 변환"""
    major_data = analysis.get("major", {})
    extra_data = analysis.get("extra", {})
    meta_raw = extra_data.get("meta_data", {})

    major = MajorAttributes(
        category=major_data.get("category", "ETC"),
        color=major_data.get("color", []),
        material=major_data.get("material", []),
        style_tags=major_data.get("style_tags", []),
    )
    extra = ExtraAttributes(
        meta_data=ExtraMetadata(
            gender=meta_raw.get("gender"),
            season=meta_raw.get("season", []),
            formality=meta_raw.get("formality"),
            fit=meta_raw.get("fit"),
            occasion=meta_raw.get("occasion", []),
        ),
        caption=extra_data.get("caption"),
    )
    return major, extra


def build_commerce_payload(
    product: dict,
    major: MajorAttributes,
    extra: ExtraAttributes,
    embedding_text: str,
) -> dict:
    """네이버 메타 + AI 분석 결과 → Qdrant payload 구성
    (closet의 EmbeddingService._build_payload와 동일한 역할)
    """
    return {
        # 네이버 쇼핑 원본 데이터
        "productId": product["productId"],
        "title": product["title"],
        "link": product["link"],
        "imageUrl": product["imageUrl"],
        "price": product["price"],
        "mallName": product["mallName"],
        "brand": product["brand"],
        "maker": product["maker"],
        "category1": product["category1"],
        "category2": product["category2"],
        "category3": product["category3"],
        "category4": product["category4"],
        "searchKeyword": product["searchKeyword"],
        # AI 분석 결과
        "Category": major.category,
        "color": major.color,
        "material": major.material,
        "styleTags": major.style_tags,
        "gender": extra.meta_data.gender,
        "season": extra.meta_data.season,
        "formality": extra.meta_data.formality,
        "fit": extra.meta_data.fit,
        "occasion": extra.meta_data.occasion,
        "caption": extra.caption,
        "embeddingText": embedding_text,
    }


async def upsert_to_qdrant(
    client: AsyncQdrantClient,
    collection: str,
    product_id: str,
    vector: list[float],
    payload: dict,
) -> None:
    """Qdrant에 포인트 하나 저장"""
    point_id = int(product_id) if product_id.isdigit() else hash(product_id) % (2**63)
    await client.upsert(
        collection_name=collection,
        points=[
            qdrant_models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        ],
    )


async def process_single_product(
    product: dict,
    analyzer: GeminiImageAnalyzer,
    formatter: HybridFormatter,
    embedding_client: UpstageEmbeddingClient,
    qdrant: AsyncQdrantClient,
    collection: str,
) -> bool:
    """상품 1개 처리: 이미지 다운 → 분석 → 임베딩 → 저장"""
    # 1. 이미지 다운로드
    image_bytes = await download_image(product["imageUrl"])

    # 2. Gemini 이미지 분석 → Pydantic 모델 변환
    analysis = await analyzer.analyze_image(image_bytes)
    major, extra = parse_analysis(analysis)

    # 3. 임베딩 텍스트 생성 + 벡터 생성
    embedding_text = formatter.format(major, extra)
    vector = await embedding_client.embed(embedding_text)

    # 4. payload 구성 + Qdrant 저장
    payload = build_commerce_payload(product, major, extra, embedding_text)
    await upsert_to_qdrant(qdrant, collection, product["productId"], vector, payload)

    return True


async def process_all_products(products: list[dict]) -> tuple[int, int]:
    """전체 상품 순차 처리 (오케스트레이션)"""
    settings = get_settings()
    analyzer = GeminiImageAnalyzer()
    formatter = HybridFormatter()
    embedding_client = UpstageEmbeddingClient()

    qdrant = await get_qdrant_client()
    collection = settings.qdrant_shop_collection_name

    success, fail = 0, 0

    for i, product in enumerate(products):
        logger.info(f"[{i + 1}/{len(products)}] 처리 중: {product['title'][:30]}...")

        try:
            await process_single_product(
                product, analyzer, formatter, embedding_client, qdrant, collection
            )
            success += 1
            logger.info("  ✅ 저장 완료")
        except Exception as e:
            fail += 1
            logger.error(f"  ❌ 실패: {e}")

        await asyncio.sleep(1.0)  # API rate limit 방지

    return success, fail


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def main() -> None:
    print("=" * 60)
    print("🛍️  커머스 상품 DB 크롤러")
    print("=" * 60)

    start_time = time.perf_counter()

    # DB 초기화 (Redis, Qdrant 등 자동 연결)
    await init_databases()

    try:
        # Step 1: 크롤링
        print("\n📦 [Step 1] 네이버 쇼핑 API 크롤링")
        print("-" * 60)
        products = await crawl_all_products()

        if not products:
            print("❌ 수집된 상품이 없습니다. API 키를 확인해주세요.")
            return

        # Step 2: 분석 + 임베딩 + 저장
        print(f"\n🤖 [Step 2] Gemini 분석 + 임베딩 + Qdrant 저장 ({len(products)}개)")
        print("-" * 60)
        success, fail = await process_all_products(products)

    finally:
        await close_databases()

    elapsed = time.perf_counter() - start_time

    # 최종 결과
    print("\n" + "=" * 60)
    print("📋 최종 결과")
    print("=" * 60)
    print(f"  크롤링: {len(products)}개 상품")
    print(f"  저장 성공: {success}개")
    print(f"  저장 실패: {fail}개")
    print(f"  총 소요 시간: {elapsed:.0f}초 ({elapsed / 60:.1f}분)")
    print("  Qdrant 컬렉션: commerce")


if __name__ == "__main__":
    asyncio.run(main())
