from collections.abc import Callable, Coroutine
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 핸들러 타입
MessageHandler = Callable[..., Coroutine[object, object, None]]
HandlerFactory = Callable[..., MessageHandler]


class ConsumerConfig(BaseModel):
    """토픽별 Kafka Consumer 설정."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    topic: str
    group_id: str
    handler_factory: HandlerFactory
    replicas: int = 1


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(
        default="clothes", alias="QDRANT_COLLECTION_NAME"
    )
    qdrant_shop_collection_name: str = Field(
        default="commerce", alias="QDRANT_SHOP_COLLECTION_NAME"
    )
    qdrant_use_https: bool = Field(default=False, alias="QDRANT_USE_HTTPS")
    qdrant_prefer_grpc: bool = Field(default=False, alias="QDRANT_PREFER_GRPC")

    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_password: str | None = Field(default=None, alias="REDIS_PASSWORD")
    redis_max_connections: int = Field(default=50, alias="REDIS_MAX_CONNECTIONS")

    hf_home: str = Field(default="./models", alias="HF_HOME")
    upstage_api_key: str | None = Field(default=None, alias="UPSTAGE_API_KEY")
    embedding_model: str = Field(default="embedding-passage", alias="EMBEDDING_MODEL")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    llm_timeout: int = Field(default=30, alias="LLM_TIMEOUT")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")

    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    vton_model: str = Field(default="gemini-3-pro-image-preview", alias="VTON_MODEL")

    naver_client_id: str | None = Field(default=None, alias="NAVER_CLIENT_ID")
    naver_client_secret: str | None = Field(default=None, alias="NAVER_CLIENT_SECRET")

    use_mock_analyzer: bool = Field(default=False, alias="USE_MOCK_ANALYZER")

    langfuse_enabled: bool = Field(default=True, alias="LANGFUSE_ENABLED")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_host: str = Field(default="http://localhost:3000", alias="LANGFUSE_HOST")

    # ── Kafka ──
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092", alias="KAFKA_BOOTSTRAP_SERVERS"
    )
    kafka_closet_request_topic: str = Field(
        default="ai.clothes.analyze.request", alias="KAFKA_CLOSET_REQUEST_TOPIC"
    )
    kafka_closet_result_topic: str = Field(
        default="ai.clothes.analyze.result", alias="KAFKA_CLOSET_RESULT_TOPIC"
    )
    kafka_analyze_consumer_group: str = Field(
        default="ai_analyze_worker_group", alias="KAFKA_ANALYZE_CONSUMER_GROUP"
    )
    kafka_max_concurrent_tasks: int = Field(
        default=50, alias="KAFKA_MAX_CONCURRENT_TASKS"
    )

    # ── Backend Internal API (presigned URL 발급용) ──
    backend_internal_url: str = Field(
        default="http://15.164.36.40", alias="BACKEND_INTERNAL_URL"
    )
    backend_internal_api_key: str = Field(default="", alias="BACKEND_INTERNAL_API_KEY")


@lru_cache
def get_settings() -> Settings:
    return Settings()
