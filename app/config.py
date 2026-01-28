from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    qdrant_host: str = Field(alias="QDRANT_HOST")
    qdrant_port: int = Field(alias="QDRANT_PORT")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(alias="QDRANT_COLLECTION_NAME")
    qdrant_use_https: bool = Field(default=True, alias="QDRANT_USE_HTTPS")
    qdrant_prefer_grpc: bool = Field(default=False, alias="QDRANT_PREFER_GRPC")

    redis_host: str = Field(alias="REDIS_HOST")
    redis_port: int = Field(alias="REDIS_PORT")
    redis_db: int = Field(alias="REDIS_DB")
    redis_password: str | None = Field(default=None, alias="REDIS_PASSWORD")
    redis_max_connections: int = Field(default=10, alias="REDIS_MAX_CONNECTIONS")

    aws_access_key_id: str | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(
        default=None, alias="AWS_SECRET_ACCESS_KEY"
    )
    aws_region: str = Field(default="ap-northeast-2", alias="AWS_REGION")
    s3_bucket_name: str = Field(default="klosetlab-ai-storage", alias="S3_BUCKET_NAME")

    hf_home: str = Field(default="./models", alias="HF_HOME")
    upstage_api_key: str | None = Field(default=None, alias="UPSTAGE_API_KEY")
    embedding_model: str = Field(default="embedding-passage", alias="EMBEDDING_MODEL")
    caption_model: str = Field(
        default="Salesforce/blip-image-captioning-base", alias="CAPTION_MODEL"
    )

    # LLM Settings
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    llm_timeout: int = Field(default=30, alias="LLM_TIMEOUT")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")


@lru_cache
def get_settings() -> Settings:
    return Settings()
