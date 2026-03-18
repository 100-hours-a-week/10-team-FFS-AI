from prometheus_client import Counter, Gauge, Histogram

KAFKA_MESSAGE_PROCESSING_SECONDS = Histogram(
    "kafka_message_processing_seconds",
    "Time spent processing Kafka messages (seconds)",
    ["worker_type", "topic"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, float("inf")),
)


KAFKA_MESSAGES_PROCESSED_TOTAL = Counter(
    "kafka_messages_processed_total",
    "Total number of processed Kafka messages",
    ["worker_type", "topic", "status"],
)


KAFKA_CONSUMER_LAG = Gauge(
    "kafka_consumer_lag",
    "Number of unprocessed messages in Kafka topic",
    ["worker_type", "topic", "partition"],
)


KAFKA_DLQ_MESSAGES_TOTAL = Counter(
    "kafka_dlq_messages_total",
    "Total number of messages sent to DLQ",
    ["worker_type", "original_topic", "error_type"],
)

# 재시도 횟수
KAFKA_RETRY_TOTAL = Counter(
    "kafka_retry_total",
    "Total number of message retries",
    ["worker_type", "topic", "retry_number"],
)

RATE_LIMIT_WAIT_SECONDS = Histogram(
    "rate_limit_wait_seconds",
    "Time spent waiting for rate limit token (seconds)",
    ["key"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, float("inf")),
)


RATE_LIMIT_HITS_TOTAL = Counter(
    "rate_limit_hits_total",
    "Total number of rate limit hits (had to wait)",
    ["key"],
)
