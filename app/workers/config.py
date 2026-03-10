from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"

    OUTFIT_GROUP_ID: str = "outfit-consumer-group"
    OUTFIT_REQUEST_TOPIC: str = "outfit-request"
    OUTFIT_RESPONSE_TOPIC: str = "outfit-response"

    SHOP_GROUP_ID: str = "shop-consumer-group"
    SHOP_REQUEST_TOPIC: str = "shop-request"
    SHOP_RESPONSE_TOPIC: str = "shop-response"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


worker_settings = WorkerSettings()
