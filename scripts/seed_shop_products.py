"""쇼핑 상품 시드 스크립트

commerce 컬렉션을 생성하고 무신사 스타일 가상 상품 데이터를 삽입합니다.

사용법:
    .venv/bin/python scripts/seed_shop_products.py
"""

import asyncio
import logging
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Upstage Embedding 벡터 차원 수
VECTOR_SIZE = 4096

# ============================================================
# 무신사 스타일 가상 상품 데이터
# ============================================================
SHOP_PRODUCTS = [
    # ── TOP (상의) ──────────────────────────────────
    {
        "productId": "mss_top_001",
        "title": "오버핏 피그먼트 워싱 반팔티",
        "brand": "무신사스탠다드",
        "price": 19900,
        "imageUrl": "https://image.musinsa.com/images/goods_img/2024/mss_top_001.jpg",
        "link": "https://www.musinsa.com/app/goods/mss_top_001",
        "source": "musinsa",
        "category": "TOP",
        "styleTags": ["캐주얼", "오버핏", "베이직"],
        "caption": "편안한 오버핏 실루엣의 피그먼트 워싱 반팔 티셔츠입니다.",
    },
    {
        "productId": "mss_top_002",
        "title": "Y2K 빈티지 크롭 니트",
        "brand": "키르시",
        "price": 29000,
        "imageUrl": "https://image.musinsa.com/images/goods_img/2024/mss_top_002.jpg",
        "link": "https://www.musinsa.com/app/goods/mss_top_002",
        "source": "musinsa",
        "category": "TOP",
        "styleTags": ["Y2K", "크롭", "빈티지", "여성"],
        "caption": "Y2K 감성의 빈티지 크롭 니트. 하이웨이스트 하의와 매칭하면 레트로한 무드를 연출할 수 있습니다.",
    },
    {
        "productId": "mss_top_003",
        "title": "옥스포드 버튼다운 셔츠",
        "brand": "무신사스탠다드",
        "price": 34900,
        "imageUrl": "https://image.musinsa.com/images/goods_img/2024/mss_top_003.jpg",
        "link": "https://www.musinsa.com/app/goods/mss_top_003",
        "source": "musinsa",
        "category": "TOP",
        "styleTags": ["비즈캐주얼", "클래식", "셔츠"],
        "caption": "깔끔한 핏감의 옥스포드 버튼다운 셔츠. 면접이나 비즈니스 캐주얼에 제격입니다.",
    },
    {
        "productId": "mss_top_004",
        "title": "스트라이프 오픈카라 반팔셔츠",
        "brand": "커버낫",
        "price": 49000,
        "imageUrl": "https://image.musinsa.com/images/goods_img/2024/mss_top_004.jpg",
        "link": "https://www.musinsa.com/app/goods/mss_top_004",
        "source": "musinsa",
        "category": "TOP",
        "styleTags": ["캐주얼", "여름", "스트라이프"],
        "caption": "시원한 스트라이프 패턴의 오픈카라 셔츠. 여름 데이트나 휴양지에 잘 어울립니다.",
    },
    {
        "productId": "mss_top_005",
        "title": "헤비웨이트 그래픽 맨투맨",
        "brand": "디스이즈네버댓",
        "price": 59000,
        "imageUrl": "https://image.musinsa.com/images/goods_img/2024/mss_top_005.jpg",
        "link": "https://www.musinsa.com/app/goods/mss_top_005",
        "source": "musinsa",
        "category": "TOP",
        "styleTags": ["스트릿", "그래픽", "오버핏"],
        "caption": "두꺼운 원단의 헤비웨이트 그래픽 맨투맨. 스트릿 감성을 원하는 분에게 추천합니다.",
    },
    # ── BOTTOM (하의) ──────────────────────────────
    {
        "productId": "mss_btm_001",
        "title": "와이드핏 데님 팬츠",
        "brand": "무신사스탠다드",
        "price": 39900,
        "imageUrl": "https://image.musinsa.com/images/goods_img/2024/mss_btm_001.jpg",
        "link": "https://www.musinsa.com/app/goods/mss_btm_001",
        "source": "musinsa",
        "category": "BOTTOM",
        "styleTags": ["캐주얼", "와이드핏", "데님"],
        "caption": "편안한 와이드핏 실루엣의 데님 팬츠. 어떤 상의와도 무난하게 매칭됩니다.",
    },
    {
        "productId": "mss_btm_002",
        "title": "로우라이즈 와이드 데님",
        "brand": "키르시",
        "price": 45000,
        "imageUrl": "https://image.musinsa.com/images/goods_img/2024/mss_btm_002.jpg",
        "link": "https://www.musinsa.com/app/goods/mss_btm_002",
        "source": "musinsa",
        "category": "BOTTOM",
        "styleTags": ["Y2K", "로우라이즈", "와이드", "여성"],
        "caption": "Y2K 감성의 로우라이즈 와이드 데님. 크롭탑과 매칭하면 레트로 무드를 완성합니다.",
    },
    {
        "productId": "mss_btm_003",
        "title": "테이퍼드 슬랙스",
        "brand": "무신사스탠다드",
        "price": 29900,
        "imageUrl": "https://image.musinsa.com/images/goods_img/2024/mss_btm_003.jpg",
        "link": "https://www.musinsa.com/app/goods/mss_btm_003",
        "source": "musinsa",
        "category": "BOTTOM",
        "styleTags": ["비즈캐주얼", "포멀", "슬랙스"],
        "caption": "깔끔한 테이퍼드 핏의 슬랙스. 면접이나 포멀한 자리에 적합합니다.",
    },
    {
        "productId": "mss_btm_004",
        "title": "카고 조거 팬츠",
        "brand": "디스이즈네버댓",
        "price": 69000,
        "imageUrl": "https://image.musinsa.com/images/goods_img/2024/mss_btm_004.jpg",
        "link": "https://www.musinsa.com/app/goods/mss_btm_004",
        "source": "musinsa",
        "category": "BOTTOM",
        "styleTags": ["스트릿", "카고", "조거"],
        "caption": "스트릿 감성의 카고 조거 팬츠. 대용량 포켓이 포인트입니다.",
    },
    {
        "productId": "mss_btm_005",
        "title": "코튼 밴딩 쇼츠",
        "brand": "무신사스탠다드",
        "price": 19900,
        "imageUrl": "https://image.musinsa.com/images/goods_img/2024/mss_btm_005.jpg",
        "link": "https://www.musinsa.com/app/goods/mss_btm_005",
        "source": "musinsa",
        "category": "BOTTOM",
        "styleTags": ["캐주얼", "여름", "쇼츠"],
        "caption": "가볍고 편한 코튼 밴딩 쇼츠. 여름 일상복으로 최적입니다.",
    },
    # ── SHOES (신발) ──────────────────────────────
    {
        "productId": "mss_shoes_001",
        "title": "캔버스 로우 스니커즈",
        "brand": "컨버스",
        "price": 55000,
        "imageUrl": "https://image.musinsa.com/images/goods_img/2024/mss_shoes_001.jpg",
        "link": "https://www.musinsa.com/app/goods/mss_shoes_001",
        "source": "musinsa",
        "category": "SHOES",
        "styleTags": ["캐주얼", "스니커즈", "클래식"],
        "caption": "클래식한 캔버스 로우 스니커즈. 어떤 스타일에도 잘 어울리는 만능 아이템입니다.",
    },
    {
        "productId": "mss_shoes_002",
        "title": "에어포스 1 '07 화이트",
        "brand": "나이키",
        "price": 139000,
        "imageUrl": "https://image.musinsa.com/images/goods_img/2024/mss_shoes_002.jpg",
        "link": "https://www.musinsa.com/app/goods/mss_shoes_002",
        "source": "musinsa",
        "category": "SHOES",
        "styleTags": ["캐주얼", "스트릿", "화이트"],
        "caption": "나이키 에어포스 1 화이트. 스트릿부터 캐주얼까지 활용도 만점입니다.",
    },
    {
        "productId": "mss_shoes_003",
        "title": "더비슈즈 클래식 블랙",
        "brand": "닥터마틴",
        "price": 189000,
        "imageUrl": "https://image.musinsa.com/images/goods_img/2024/mss_shoes_003.jpg",
        "link": "https://www.musinsa.com/app/goods/mss_shoes_003",
        "source": "musinsa",
        "category": "SHOES",
        "styleTags": ["포멀", "클래식", "더비슈즈"],
        "caption": "포멀한 자리에도 잘 어울리는 클래식 블랙 더비슈즈입니다.",
    },
    {
        "productId": "mss_shoes_004",
        "title": "런닝화 프레쉬폼 X",
        "brand": "뉴발란스",
        "price": 99000,
        "imageUrl": "https://image.musinsa.com/images/goods_img/2024/mss_shoes_004.jpg",
        "link": "https://www.musinsa.com/app/goods/mss_shoes_004",
        "source": "musinsa",
        "category": "SHOES",
        "styleTags": ["스포츠", "런닝", "캐주얼"],
        "caption": "뉴발란스 프레쉬폼 런닝화. 데일리 캐주얼로도, 운동용으로도 좋습니다.",
    },
    {
        "productId": "mss_shoes_005",
        "title": "청키 플랫폼 로퍼",
        "brand": "키르시",
        "price": 65000,
        "imageUrl": "https://image.musinsa.com/images/goods_img/2024/mss_shoes_005.jpg",
        "link": "https://www.musinsa.com/app/goods/mss_shoes_005",
        "source": "musinsa",
        "category": "SHOES",
        "styleTags": ["Y2K", "플랫폼", "로퍼", "여성"],
        "caption": "Y2K 감성의 청키 플랫폼 로퍼. 키높이 효과와 트렌디함을 동시에.",
    },
    # ── DRESS (원피스) ──────────────────────────────
    {
        "productId": "mss_dress_001",
        "title": "플로럴 미디 원피스",
        "brand": "마리끌레르",
        "price": 79000,
        "imageUrl": "https://image.musinsa.com/images/goods_img/2024/mss_dress_001.jpg",
        "link": "https://www.musinsa.com/app/goods/mss_dress_001",
        "source": "musinsa",
        "category": "DRESS",
        "styleTags": ["로맨틱", "플로럴", "미디", "여성"],
        "caption": "화사한 플로럴 패턴의 미디 원피스. 봄 데이트에 제격입니다.",
    },
    {
        "productId": "mss_dress_002",
        "title": "린넨 셔츠 원피스",
        "brand": "무신사스탠다드",
        "price": 49900,
        "imageUrl": "https://image.musinsa.com/images/goods_img/2024/mss_dress_002.jpg",
        "link": "https://www.musinsa.com/app/goods/mss_dress_002",
        "source": "musinsa",
        "category": "DRESS",
        "styleTags": ["캐주얼", "린넨", "여름", "여성"],
        "caption": "시원한 린넨 소재의 셔츠 원피스. 여름 날씨에 편하게 입기 좋습니다.",
    },
    # ── ACCESSORY (악세서리) ──────────────────────
    {
        "productId": "mss_acc_001",
        "title": "빈티지 볼캡",
        "brand": "디스이즈네버댓",
        "price": 35000,
        "imageUrl": "https://image.musinsa.com/images/goods_img/2024/mss_acc_001.jpg",
        "link": "https://www.musinsa.com/app/goods/mss_acc_001",
        "source": "musinsa",
        "category": "ACCESSORY",
        "styleTags": ["스트릿", "볼캡", "빈티지"],
        "caption": "빈티지 워싱 가공의 볼캡. 스트릿 코디의 마무리 포인트입니다.",
    },
    {
        "productId": "mss_acc_002",
        "title": "미니멀 가죽 토트백",
        "brand": "마르헨제이",
        "price": 89000,
        "imageUrl": "https://image.musinsa.com/images/goods_img/2024/mss_acc_002.jpg",
        "link": "https://www.musinsa.com/app/goods/mss_acc_002",
        "source": "musinsa",
        "category": "ACCESSORY",
        "styleTags": ["미니멀", "가죽", "토트백", "여성"],
        "caption": "미니멀한 디자인의 가죽 토트백. 출근 코디에 잘 어울립니다.",
    },
]


async def embed_text(text: str, api_key: str, model: str) -> list[float]:
    """Upstage Solar Embedding API로 텍스트를 벡터로 변환"""
    url = "https://api.upstage.ai/v1/solar/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "input": text}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url, headers=headers, json=payload, timeout=15.0
        )
        response.raise_for_status()
        result = response.json()
        return result["data"][0]["embedding"]


def build_embedding_text(product: dict) -> str:
    """상품 정보를 임베딩용 텍스트로 조합

    EmbeddingTextFormatter(HybridFormatter)의 로직을 참고하여
    카테고리, 스타일, 캡션 등을 결합합니다.
    """
    parts = []

    # 카테고리
    parts.append(f"카테고리: {product['category']}")

    # 스타일 태그
    if product.get("styleTags"):
        parts.append(f"스타일: {', '.join(product['styleTags'])}")

    # 브랜드
    if product.get("brand"):
        parts.append(f"브랜드: {product['brand']}")

    # 캡션 (가장 중요한 정보)
    if product.get("caption"):
        parts.append(product["caption"])

    return " | ".join(parts)


async def main() -> None:
    settings = get_settings()

    # API 키 확인
    if not settings.upstage_api_key:
        logger.error("❌ UPSTAGE_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    collection_name = settings.qdrant_shop_collection_name
    logger.info(f"🎯 대상 컬렉션: {collection_name}")
    logger.info(f"📦 상품 수: {len(SHOP_PRODUCTS)}개")

    # Qdrant 연결
    qdrant = AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        https=settings.qdrant_use_https,
        timeout=10,
    )

    # 컬렉션 존재 여부 확인 및 생성
    collections = await qdrant.get_collections()
    collection_names = [c.name for c in collections.collections]

    if collection_name in collection_names:
        logger.info(f"⚠️  컬렉션 '{collection_name}' 이미 존재합니다. 삭제 후 재생성합니다.")
        await qdrant.delete_collection(collection_name)

    logger.info(f"📌 컬렉션 '{collection_name}' 생성 중...")
    await qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=VECTOR_SIZE,
            distance=qdrant_models.Distance.COSINE,
        ),
    )

    # 페이로드 인덱스 생성 (필터 성능 최적화)
    for field_name, field_type in [
        ("category", qdrant_models.PayloadSchemaType.KEYWORD),
        ("brand", qdrant_models.PayloadSchemaType.KEYWORD),
        ("price", qdrant_models.PayloadSchemaType.INTEGER),
    ]:
        await qdrant.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_type,
        )
        logger.info(f"  📎 인덱스 생성: {field_name} ({field_type})")

    # 상품 임베딩 및 upsert
    points = []
    for idx, product in enumerate(SHOP_PRODUCTS, start=1):
        embedding_text = build_embedding_text(product)
        logger.info(
            f"  [{idx}/{len(SHOP_PRODUCTS)}] 임베딩 중: "
            f"{product['title']} ({product['category']})"
        )

        vector = await embed_text(
            embedding_text,
            settings.upstage_api_key,
            settings.embedding_model,
        )

        points.append(
            qdrant_models.PointStruct(
                id=idx,
                vector=vector,
                payload=product,
            )
        )

    # 배치 upsert
    logger.info(f"💾 {len(points)}개 포인트 upsert 중...")
    await qdrant.upsert(
        collection_name=collection_name,
        points=points,
    )

    # 결과 확인
    collection_info = await qdrant.get_collection(collection_name)
    logger.info(
        f"\n✅ 시드 데이터 삽입 완료!\n"
        f"   컬렉션: {collection_name}\n"
        f"   포인트 수: {collection_info.points_count}\n"
        f"   벡터 차원: {VECTOR_SIZE}\n"
        f"   카테고리 분포:\n"
        f"     TOP: 5개\n"
        f"     BOTTOM: 5개\n"
        f"     SHOES: 5개\n"
        f"     DRESS: 2개\n"
        f"     ACCESSORY: 2개"
    )

    await qdrant.close()


if __name__ == "__main__":
    asyncio.run(main())
