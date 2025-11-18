"""Core data models for Conduit routing system.

This module defines Pydantic models for queries, routing decisions,
responses, feedback, and ML model state.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class QueryConstraints(BaseModel):
    """Constraints for routing decisions."""

    max_cost: float | None = Field(
        None, description="Maximum cost in dollars", ge=0.0
    )
    max_latency: float | None = Field(
        None, description="Maximum latency in seconds", ge=0.0
    )
    min_quality: float | None = Field(
        None, description="Minimum quality score (0.0-1.0)", ge=0.0, le=1.0
    )
    preferred_provider: str | None = Field(
        None, description="Preferred LLM provider (openai, anthropic, google, groq)"
    )


class Query(BaseModel):
    """User query to be routed to an LLM."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique query ID")
    text: str = Field(..., description="Query text", min_length=1)
    user_id: str | None = Field(None, description="User identifier")
    context: dict[str, Any] | None = Field(
        None, description="Additional context metadata"
    )
    constraints: QueryConstraints | None = Field(
        None, description="Routing constraints"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Query creation timestamp"
    )

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Validate text is not empty or whitespace."""
        if not v.strip():
            raise ValueError("Query text cannot be empty")
        return v.strip()


class QueryFeatures(BaseModel):
    """Extracted features from query for routing decision."""

    embedding: list[float] = Field(
        ..., description="Semantic embedding (384-dim)", min_length=384, max_length=384
    )
    token_count: int = Field(..., description="Approximate token count", ge=0)
    complexity_score: float = Field(
        ..., description="Complexity score (0.0-1.0)", ge=0.0, le=1.0
    )
    domain: str = Field(..., description="Query domain classification")
    domain_confidence: float = Field(
        ..., description="Domain classification confidence", ge=0.0, le=1.0
    )


class RoutingDecision(BaseModel):
    """ML-powered routing decision for a query."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Decision ID")
    query_id: str = Field(..., description="Associated query ID")
    selected_model: str = Field(..., description="Selected LLM model")
    confidence: float = Field(
        ..., description="Thompson sampling confidence", ge=0.0, le=1.0
    )
    features: QueryFeatures = Field(..., description="Extracted query features")
    reasoning: str = Field(..., description="Explanation of routing decision")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Decision timestamp"
    )


class Response(BaseModel):
    """LLM response to a query."""

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique response ID"
    )
    query_id: str = Field(..., description="Associated query ID")
    model: str = Field(..., description="Model that generated response")
    text: str = Field(..., description="Response text (JSON for structured outputs)")
    cost: float = Field(..., description="Cost in dollars", ge=0.0)
    latency: float = Field(..., description="Latency in seconds", ge=0.0)
    tokens: int = Field(..., description="Total tokens used", ge=0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Response timestamp"
    )


class Feedback(BaseModel):
    """User feedback on response quality."""

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique feedback ID"
    )
    response_id: str = Field(..., description="Associated response ID")
    quality_score: float = Field(
        ..., description="Quality score (0.0-1.0)", ge=0.0, le=1.0
    )
    user_rating: int | None = Field(
        None, description="User rating (1-5 stars)", ge=1, le=5
    )
    met_expectations: bool = Field(
        ..., description="Whether response met user expectations"
    )
    comments: str | None = Field(None, description="Optional user comments")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Feedback timestamp"
    )


class ModelState(BaseModel):
    """Thompson Sampling state for a model (Beta distribution)."""

    model_id: str = Field(..., description="Model identifier")
    alpha: float = Field(default=1.0, description="Beta distribution α parameter", gt=0)
    beta: float = Field(default=1.0, description="Beta distribution β parameter", gt=0)
    total_requests: int = Field(default=0, description="Total requests to this model")
    total_cost: float = Field(default=0.0, description="Total cost accumulated", ge=0.0)
    avg_quality: float = Field(
        default=0.0, description="Average quality score", ge=0.0, le=1.0
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Last update timestamp"
    )

    @property
    def mean_success_rate(self) -> float:
        """Expected success rate (mean of Beta distribution)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Variance of Beta distribution."""
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab * ab * (ab + 1))


class RoutingResult(BaseModel):
    """Complete routing result returned to client."""

    id: str = Field(..., description="Response ID")
    query_id: str = Field(..., description="Query ID")
    model: str = Field(..., description="Model used")
    data: dict[str, Any] = Field(..., description="Structured response data")
    metadata: dict[str, Any] = Field(..., description="Routing metadata")

    @classmethod
    def from_response(
        cls, response: Response, routing: RoutingDecision
    ) -> "RoutingResult":
        """Create RoutingResult from Response and RoutingDecision."""
        import json

        return cls(
            id=response.id,
            query_id=response.query_id,
            model=response.model,
            data=json.loads(response.text),
            metadata={
                "cost": response.cost,
                "latency": response.latency,
                "tokens": response.tokens,
                "routing_confidence": routing.confidence,
                "reasoning": routing.reasoning,
            },
        )
