class EmbeddingServiceError(Exception):
    pass


class ExternalAPIError(EmbeddingServiceError):

    def __init__(self, service: str, message: str):
        self.service = service
        self.message = message
        super().__init__(f"{service} API error: {message}")


class VectorDBError(EmbeddingServiceError):

    def __init__(self, operation: str, message: str):
        self.operation = operation
        self.message = message
        super().__init__(f"VectorDB {operation} failed: {message}")
