from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    kafka_bootstrap_servers: str = Field(
        default="localhost:9092",
        alias="KAFKA_BOOTSTRAP_SERVERS",
    )

    consumer_auto_offset_reset: str = Field(
        default="earliest",
        alias="KAFKA_CONSUMER_AUTO_OFFSET_RESET",
    )
    consumer_max_poll_records: int = Field(
        default=1,
        alias="KAFKA_CONSUMER_MAX_POLL_RECORDS",
    )
    consumer_session_timeout_ms: int = Field(
        default=30000,
        alias="KAFKA_CONSUMER_SESSION_TIMEOUT_MS",
    )
    consumer_heartbeat_interval_ms: int = Field(
        default=10000,
        alias="KAFKA_CONSUMER_HEARTBEAT_INTERVAL_MS",
    )

    max_concurrent_tasks: int = Field(
        default=50,
        alias="KAFKA_MAX_CONCURRENT_TASKS",
    )

    shutdown_timeout_seconds: int = Field(
        default=30,
        alias="WORKER_SHUTDOWN_TIMEOUT_SECONDS",
    )


class OutfitWorkerConfig(WorkerConfig):
    request_topic: str = Field(
        default="outfit-request",
        alias="KAFKA_OUTFIT_REQUEST_TOPIC",
    )
    response_topic: str = Field(
        default="outfit-response",
        alias="KAFKA_OUTFIT_RESPONSE_TOPIC",
    )
    dlq_topic: str = Field(
        default="outfit-dlq",
        alias="KAFKA_OUTFIT_DLQ_TOPIC",
    )

    group_id: str = Field(
        default="outfit-consumer-group",
        alias="KAFKA_OUTFIT_CONSUMER_GROUP",
    )


class ShopWorkerConfig(WorkerConfig):
    request_topic: str = Field(
        default="shop-request",
        alias="KAFKA_SHOP_REQUEST_TOPIC",
    )
    response_topic: str = Field(
        default="shop-response",
        alias="KAFKA_SHOP_RESPONSE_TOPIC",
    )
    dlq_topic: str = Field(
        default="shop-dlq",
        alias="KAFKA_SHOP_DLQ_TOPIC",
    )

    group_id: str = Field(
        default="shop-consumer-group",
        alias="KAFKA_SHOP_CONSUMER_GROUP",
    )


@lru_cache
def get_outfit_worker_config() -> OutfitWorkerConfig:
    return OutfitWorkerConfig()


@lru_cache
def get_shop_worker_config() -> ShopWorkerConfig:
    return ShopWorkerConfig()
