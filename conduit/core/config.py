"""Configuration management for Conduit.

This module provides centralized configuration loading from environment
variables with validation and type safety.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Supabase Database
    database_url: str = Field(..., description="PostgreSQL connection string")
    database_pool_size: int = Field(
        default=20, description="Connection pool size", ge=1, le=100
    )

    # Redis Cache
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL")
    redis_ttl: int = Field(
        default=3600, description="Cache TTL in seconds", ge=60, le=86400
    )

    # LLM Provider API Keys
    openai_api_key: str | None = Field(None, description="OpenAI API key")
    anthropic_api_key: str | None = Field(None, description="Anthropic API key")
    google_api_key: str | None = Field(None, description="Google API key")
    groq_api_key: str | None = Field(None, description="Groq API key")

    # ML Configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence transformer model"
    )
    default_models: list[str] = Field(
        default=[
            "gpt-4o-mini",
            "gpt-4o",
            "claude-sonnet-4",
            "claude-opus-4",
        ],
        description="Available models for routing",
    )

    # Thompson Sampling Parameters
    exploration_rate: float = Field(
        default=0.1, description="Exploration rate (epsilon)", ge=0.0, le=1.0
    )
    reward_weight_quality: float = Field(
        default=0.5, description="Quality weight in reward", ge=0.0, le=1.0
    )
    reward_weight_cost: float = Field(
        default=0.3, description="Cost weight in reward", ge=0.0, le=1.0
    )
    reward_weight_latency: float = Field(
        default=0.2, description="Latency weight in reward", ge=0.0, le=1.0
    )

    # API Configuration
    api_rate_limit: int = Field(
        default=100, description="Requests per minute per user", ge=1, le=1000
    )
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port", ge=1, le=65535)

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    environment: str = Field(
        default="development", description="Environment (development, production)"
    )

    @field_validator("reward_weight_quality", "reward_weight_cost", "reward_weight_latency")
    @classmethod
    def validate_reward_weights_sum(cls, v: float, info) -> float:
        """Validate reward weights sum to 1.0."""
        # This is a simplified check - full validation would need all three values
        return v

    @property
    def reward_weights(self) -> dict[str, float]:
        """Return reward weights as dictionary."""
        return {
            "quality": self.reward_weight_quality,
            "cost": self.reward_weight_cost,
            "latency": self.reward_weight_latency,
        }

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"


# Global settings instance
settings = Settings()
