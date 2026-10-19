"""Custom exceptions used across the application."""


class ApplicationError(Exception):
    """Base exception for application-level failures."""


class ConfigurationError(ApplicationError):
    """Raised when required application configuration is missing or invalid."""


class DatabaseConnectionError(ApplicationError):
    """Raised when a database connectivity issue occurs."""


class VectorStoreConnectionError(ApplicationError):
    """Raised when the vector store cannot be reached or initialized."""


class ValidationError(ApplicationError):
    """Raised when input validation fails."""
