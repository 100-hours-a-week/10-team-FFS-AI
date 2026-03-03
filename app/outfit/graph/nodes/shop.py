import logging

from langgraph.types import RunnableConfig

from app.outfit.graph.nodes.search import get_min_count
from app.outfit.graph.state import OutfitGraphState
from app.outfit.schemas import ClothingCandidate, ParsedQuery, SearchResult
from app.shop.schemas import (
    ProductCandidate,
    ProductSearchResult,
    ShopParsedQuery,
    ShopSearchQuery,
)

logger = logging.getLogger(__name__)


def to_shop_parsed_query(parsed_query: ParsedQuery) -> ShopParsedQuery:
    return ShopParsedQuery(
        occasion=parsed_query.occasion,
        style=parsed_query.style,
        season=parsed_query.season,
        price_max=None,
        price_min=None,
        brand=None,
        target_category=None,
        constraints=getattr(parsed_query, "constraints", []),
    )


def convert_product_to_clothing(product: ProductCandidate) -> ClothingCandidate:
    return ClothingCandidate(
        clothes_id=abs(hash(product.product_id)) % (10**9),
        image_url=product.image_url,
        category=product.category,
        color=[],
        style_tags=product.style_tags,
        caption=product.title,
        similarity_score=product.similarity_score,
        source="shop",
        price=product.price,
        brand=product.brand if product.brand else None,
        purchase_url=product.link,
    )


def convert_to_search_results(
    shop_results: list[ProductSearchResult],
) -> list[SearchResult]:
    return [
        SearchResult(
            category=result.category,
            candidates=[convert_product_to_clothing(p) for p in result.candidates],
        )
        for result in shop_results
    ]


async def supplement_from_shop(state: OutfitGraphState, config: RunnableConfig) -> dict:
    configurable = config.get("configurable", {})
    shop_repository = configurable["shop_repository"]
    shop_search_builder = configurable["shop_search_builder"]

    parsed_query = state["parsed_query"]
    search_results = state.get("search_results", [])
    required_categories = state.get("required_categories", [])
    category_coverage = state.get("category_coverage", {})
    trace_id = state.get("trace_id", "")

    missing_categories = [
        cat
        for cat in required_categories
        if category_coverage.get(cat, 0) < get_min_count(cat)
    ]

    logger.info(
        f"Supplementing from shop | trace_id={trace_id} "
        f"missing_categories={missing_categories}"
    )

    shop_parsed = to_shop_parsed_query(parsed_query)

    all_shop_queries: list[ShopSearchQuery] = []
    for cat in missing_categories:
        shop_parsed_for_cat = ShopParsedQuery(
            occasion=shop_parsed.occasion,
            style=shop_parsed.style,
            season=shop_parsed.season,
            price_max=None,
            price_min=None,
            brand=None,
            target_category=cat,
            constraints=shop_parsed.constraints,
        )
        queries = shop_search_builder.build(shop_parsed_for_cat)
        all_shop_queries.extend(queries)

    shop_results: list[ProductSearchResult] = []
    if all_shop_queries:
        shop_results = await shop_repository.search_multiple(
            queries=all_shop_queries,
            parsed=shop_parsed,
            trace_id=trace_id,
        )

    shop_search_results = convert_to_search_results(shop_results)
    merged = list(search_results) + shop_search_results

    total_shop = sum(len(r.candidates) for r in shop_search_results)
    logger.info(
        f"Shop supplement complete | trace_id={trace_id} "
        f"shop_candidates={total_shop} "
        f"total_merged={sum(len(r.candidates) for r in merged)}"
    )

    return {
        "shop_results": shop_search_results,
        "merged_candidates": merged,
        "shop_supplemented": True,
    }
