class EmbeddingServiceError(Exception):
    pass


class ExternalAPIError(EmbeddingServiceError):
    def __init__(self: "ExternalAPIError", service: str, message: str) -> None:
        self.service = service
        self.message = message
        super().__init__(f"{service} API error: {message}")


class VectorDBError(EmbeddingServiceError):
    def __init__(self: "VectorDBError", operation: str, message: str) -> None:
        self.operation = operation
        self.message = message
        super().__init__(f"VectorDB {operation} failed: {message}")
