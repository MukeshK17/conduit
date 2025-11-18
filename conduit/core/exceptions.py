"""Exception hierarchy for Conduit routing system.

This module defines custom exceptions for different failure modes
in the routing pipeline.
"""


class ConduitError(Exception):
    """Base exception for all Conduit errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AnalysisError(ConduitError):
    """Query analysis failed (embedding, complexity, domain classification)."""


class RoutingError(ConduitError):
    """Model selection failed (Thompson Sampling, constraint satisfaction)."""


class ExecutionError(ConduitError):
    """LLM execution failed (API call, timeout, parsing)."""


class DatabaseError(ConduitError):
    """Database operation failed (connection, query, transaction)."""


class ValidationError(ConduitError):
    """Input validation failed (Pydantic, constraints, schema)."""


class ConfigurationError(ConduitError):
    """Configuration error (missing env vars, invalid settings)."""


class CircuitBreakerOpenError(ConduitError):
    """Circuit breaker is open, preventing execution."""


class RateLimitError(ConduitError):
    """Rate limit exceeded for user or API."""
