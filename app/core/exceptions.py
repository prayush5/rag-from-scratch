class AppException(Exception):
    """Base exception for application-level errors."""


class RAGServiceError(AppException):
    """Base exception for RAG-related failures."""


class IngestionError(AppException):
    """Raised when document ingestion fails."""


class RetrievalError(RAGServiceError):
    """Raised when document retrieval fails."""


class GenerationError(RAGServiceError):
    """Raised when LLM generation fails."""


class ServiceUnavailableError(AppException):
    """Raised when an external dependency is unavailable."""