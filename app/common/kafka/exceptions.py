from __future__ import annotations


class KafkaWorkerError(Exception):
    pass


class PermanentError(KafkaWorkerError):
    pass


class DeserializationError(PermanentError):
    def __init__(
        self,
        message: str,
        original_data: bytes,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.original_data = original_data
        self.cause = cause

    def __str__(self) -> str:
        cause_info = f" (원인: {self.cause})" if self.cause else ""
        return f"{self.args[0]}{cause_info}"


class RetryableError(KafkaWorkerError):
    pass


class InfrastructureError(RetryableError):
    def __init__(
        self,
        message: str,
        service: str,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.service = service
        self.cause = cause

    def __str__(self) -> str:
        cause_info = f" (원인: {self.cause})" if self.cause else ""
        return f"[{self.service}] {self.args[0]}{cause_info}"


class RateLimitError(RetryableError):
    def __init__(
        self,
        message: str,
        retry_after: int,
        service: str,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.service = service

    def __str__(self) -> str:
        return f"[{self.service}] {self.args[0]} (retry_after={self.retry_after}초)"


class MaxRetriesExceededError(KafkaWorkerError):
    def __init__(
        self,
        message: str,
        retry_count: int,
        last_error: Exception,
    ) -> None:
        super().__init__(message)
        self.retry_count = retry_count
        self.last_error = last_error

    def __str__(self) -> str:
        return (
            f"{self.args[0]} "
            f"(retry_count={self.retry_count}, "
            f"last_error={self.last_error.__class__.__name__})"
        )
