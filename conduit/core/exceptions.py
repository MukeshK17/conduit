"""Exception hierarchy for Conduit routing system.

This module defines custom exceptions for different failure modes
in the routing pipeline.
"""

from typing import Any


class ConduitError(Exception):
    """Base exception for all Conduit errors."""

    code: str = "CONDUIT_ERROR"

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AnalysisError(ConduitError):
    """Query analysis failed (embedding, complexity, domain classification)."""

    code = "ANALYSIS_FAILED"


class RoutingError(ConduitError):
    """Model selection failed (Thompson Sampling, constraint satisfaction)."""

    code = "ROUTING_FAILED"


class ExecutionError(ConduitError):
    """LLM execution failed (API call, timeout, parsing)."""

    code = "EXECUTION_FAILED"


class DatabaseError(ConduitError):
    """Database operation failed (connection, query, transaction)."""

    code = "DATABASE_ERROR"


class ValidationError(ConduitError):
    """Input validation failed (Pydantic, constraints, schema)."""

    code = "VALIDATION_ERROR"


class ConfigurationError(ConduitError):
    """Configuration error (missing env vars, invalid settings)."""

    code = "CONFIGURATION_ERROR"


class CircuitBreakerOpenError(ConduitError):
    """Circuit breaker is open, preventing execution."""

    code = "CIRCUIT_BREAKER_OPEN"


class RateLimitError(ConduitError):
    """Rate limit exceeded for user or API."""

    code = "RATE_LIMIT_EXCEEDED"
