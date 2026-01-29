import logging

from qdrant_client import AsyncQdrantClient
from redis.asyncio import ConnectionPool, Redis
from tenacity import retry, stop_after_attempt, wait_fixed

from app.config import get_settings

logger = logging.getLogger(__name__)

_qdrant_client: AsyncQdrantClient | None = None
_redis_pool: ConnectionPool | None = None
_redis_client: Redis | None = None


async def get_qdrant_client() -> AsyncQdrantClient:
    if _qdrant_client is None:
        raise RuntimeError(
            "Qdrant client is not initialized. Call init_databases() first."
        )
    return _qdrant_client


def get_redis_client() -> Redis:
    if _redis_client is None:
        raise RuntimeError(
            "Redis client is not initialized. Call init_databases() first."
        )
    return _redis_client


@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
async def init_databases() -> None:
    global _qdrant_client, _redis_pool, _redis_client

    settings = get_settings()

    try:
        logger.info(
            f"Connecting to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}"
        )
        _qdrant_client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            timeout=10,
            https=settings.qdrant_use_https,
            prefer_grpc=settings.qdrant_prefer_grpc,
        )

        from qdrant_client.http import models as qdrant_models

        collections = await _qdrant_client.get_collections()
        logger.info(
            f"Qdrant connected successfully. Collections: {len(collections.collections)}"
        )

        collection_exists = any(
            col.name == settings.qdrant_collection_name
            for col in collections.collections
        )
        if not collection_exists:
            logger.info(f"Creating collection '{settings.qdrant_collection_name}'")
            await _qdrant_client.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=4096, distance=qdrant_models.Distance.COSINE
                ),
            )
            logger.info(f"Collection '{settings.qdrant_collection_name}' created")
        else:
            logger.info(f"Collection '{settings.qdrant_collection_name}' found")

    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise

    try:
        logger.info(
            f"Connecting to Redis at {settings.redis_host}:{settings.redis_port}"
        )

        _redis_pool = ConnectionPool(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password if settings.redis_password else None,
            max_connections=settings.redis_max_connections,
            decode_responses=True,
        )

        _redis_client = Redis(connection_pool=_redis_pool)

        pong = await _redis_client.ping()
        if pong:
            logger.info(f"Redis connected successfully (DB: {settings.redis_db})")
        else:
            raise ConnectionError("Redis PING failed")

    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        if _redis_client:
            await _redis_client.close()
            _redis_client = None
        if _redis_pool:
            await _redis_pool.disconnect()
            _redis_pool = None
        raise

    logger.info("All database connections initialized successfully")


async def close_databases() -> None:
    global _qdrant_client, _redis_pool, _redis_client

    if _qdrant_client:
        try:
            await _qdrant_client.close()
            logger.info("Qdrant connection closed")
        except Exception as e:
            logger.error(f"Error closing Qdrant connection: {e}")
        finally:
            _qdrant_client = None

    if _redis_client:
        try:
            await _redis_client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
        finally:
            _redis_client = None

    if _redis_pool:
        try:
            await _redis_pool.disconnect()
            logger.info("Redis connection pool closed")
        except Exception as e:
            logger.error(f"Error closing Redis pool: {e}")
        finally:
            _redis_pool = None


async def check_health() -> dict[str, str]:
    health_status = {}

    try:
        if _qdrant_client:
            await _qdrant_client.get_collections()
            health_status["qdrant"] = "connected"
        else:
            health_status["qdrant"] = "not_initialized"
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        health_status["qdrant"] = f"error: {str(e)}"

    try:
        if _redis_client:
            await _redis_client.ping()
            health_status["redis"] = "connected"
        else:
            health_status["redis"] = "not_initialized"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        health_status["redis"] = f"error: {str(e)}"

    return health_status
