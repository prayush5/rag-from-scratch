class AppException(Exception):
    """Base exception for application-level errors."""
    status_code: int = 500
    message: str = "Internal application error"

    def __init__(self, message: str | None = None):
            super().__init__(message or self.message)
            self.message = message or self.message


class RAGServiceError(AppException):
    """Base exception for RAG-related failures."""
    status_code: int = 500

class IngestionError(AppException):
    """Raised when document ingestion fails."""
    status_code: int = 500

class RetrievalError(RAGServiceError):
    """Raised when document retrieval fails."""
    status_code: int = 502

class GenerationError(RAGServiceError):
    """Raised when LLM generation fails."""
    status_code: int = 502

class ServiceUnavailableError(AppException):
    """Raised when an external dependency is unavailable."""
    status_code: int = 503

class SafetyViolationError(AppException):
    """Raised when request violates safety policy"""
    status_code: int = 400