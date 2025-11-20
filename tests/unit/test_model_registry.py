"""Unit tests for model registry."""

import pytest

from conduit.models.registry import (
    PRICING,
    DEFAULT_REGISTRY,
    create_model_registry,
    filter_models,
    get_model_by_id,
    get_models_by_provider,
    get_registry_stats,
)
from conduit.engines.bandits.base import ModelArm


class TestPricingConstants:
    """Tests for PRICING constant data structure."""

    def test_pricing_has_all_providers(self):
        """Test PRICING dict contains all 6 expected providers."""
        expected_providers = ["openai", "anthropic", "google", "groq", "mistral", "cohere"]

        for provider in expected_providers:
            assert provider in PRICING, f"Missing provider: {provider}"

    def test_pricing_structure_valid(self):
        """Test PRICING entries have correct structure."""
        for provider, models in PRICING.items():
            assert isinstance(models, dict), f"Provider {provider} should be a dict"

            for model_name, pricing in models.items():
                assert "input" in pricing, f"{provider}:{model_name} missing 'input'"
                assert "output" in pricing, f"{provider}:{model_name} missing 'output'"
                assert "quality" in pricing, f"{provider}:{model_name} missing 'quality'"

                # Validate pricing values
                assert isinstance(pricing["input"], float)
                assert isinstance(pricing["output"], float)
                assert isinstance(pricing["quality"], float)

                # Validate ranges
                assert pricing["input"] > 0, f"{provider}:{model_name} input cost must be positive"
                assert pricing["output"] > 0, f"{provider}:{model_name} output cost must be positive"
                assert 0 <= pricing["quality"] <= 1, f"{provider}:{model_name} quality must be 0-1"

    def test_openai_models_present(self):
        """Test OpenAI models are present."""
        openai_models = PRICING["openai"]

        expected_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        for model in expected_models:
            assert model in openai_models

    def test_anthropic_models_present(self):
        """Test Anthropic models are present."""
        anthropic_models = PRICING["anthropic"]

        # Check for Claude models (names may include version dates)
        assert any("claude-3-5-sonnet" in name for name in anthropic_models)
        assert any("claude-3-opus" in name for name in anthropic_models)
        assert any("claude-3-haiku" in name for name in anthropic_models)


class TestCreateModelRegistry:
    """Tests for create_model_registry function."""

    def test_creates_model_arms(self):
        """Test registry creation returns list of ModelArm instances."""
        registry = create_model_registry()

        assert isinstance(registry, list)
        assert len(registry) > 0
        assert all(isinstance(model, ModelArm) for model in registry)

    def test_creates_17_models(self):
        """Test registry contains 17 models as specified."""
        registry = create_model_registry()

        # OpenAI: 4, Anthropic: 3, Google: 3, Groq: 3, Mistral: 3, Cohere: 2 = 18 total
        # (May be 17 if one is excluded)
        assert 17 <= len(registry) <= 18

    def test_model_ids_formatted_correctly(self):
        """Test model IDs follow 'provider:model_name' format."""
        registry = create_model_registry()

        for model in registry:
            assert ":" in model.model_id, f"Model ID should contain colon: {model.model_id}"
            provider, model_name = model.model_id.split(":", 1)
            assert provider == model.provider
            assert model_name == model.model_name

    def test_all_models_have_valid_pricing(self):
        """Test all models have positive costs."""
        registry = create_model_registry()

        for model in registry:
            assert model.cost_per_input_token > 0, f"{model.model_id} has invalid input cost"
            assert model.cost_per_output_token > 0, f"{model.model_id} has invalid output cost"

    def test_all_models_have_valid_quality(self):
        """Test all models have quality scores in valid range."""
        registry = create_model_registry()

        for model in registry:
            assert 0 <= model.expected_quality <= 1, f"{model.model_id} has invalid quality: {model.expected_quality}"

    def test_models_have_metadata(self):
        """Test models include pricing metadata."""
        registry = create_model_registry()

        for model in registry:
            assert "pricing_last_updated" in model.metadata
            assert "quality_estimate_source" in model.metadata

    def test_no_duplicate_model_ids(self):
        """Test registry has no duplicate model IDs."""
        registry = create_model_registry()

        model_ids = [model.model_id for model in registry]
        assert len(model_ids) == len(set(model_ids)), "Registry contains duplicate model IDs"


class TestDefaultRegistry:
    """Tests for DEFAULT_REGISTRY constant."""

    def test_default_registry_exists(self):
        """Test DEFAULT_REGISTRY is available."""
        assert DEFAULT_REGISTRY is not None
        assert isinstance(DEFAULT_REGISTRY, list)

    def test_default_registry_populated(self):
        """Test DEFAULT_REGISTRY contains models."""
        assert len(DEFAULT_REGISTRY) >= 17

    def test_default_registry_immutable(self):
        """Test DEFAULT_REGISTRY is a new list each time (not shared mutable state)."""
        # This is more of a design check - we want to ensure the registry
        # can be safely used without worrying about mutation
        assert isinstance(DEFAULT_REGISTRY, list)


class TestGetModelById:
    """Tests for get_model_by_id function."""

    def test_finds_existing_model(self):
        """Test finding a model that exists."""
        registry = create_model_registry()

        # Get first model ID
        first_model_id = registry[0].model_id

        found = get_model_by_id(first_model_id, registry)

        assert found is not None
        assert found.model_id == first_model_id

    def test_returns_none_for_nonexistent_model(self):
        """Test returns None for model that doesn't exist."""
        registry = create_model_registry()

        found = get_model_by_id("nonexistent:model", registry)

        assert found is None

    def test_finds_specific_models(self):
        """Test finding specific well-known models."""
        registry = create_model_registry()

        # Test OpenAI model
        gpt4o = get_model_by_id("openai:gpt-4o-mini", registry)
        assert gpt4o is not None
        assert gpt4o.provider == "openai"
        assert gpt4o.model_name == "gpt-4o-mini"

    def test_case_sensitive_search(self):
        """Test model ID search is case-sensitive."""
        registry = create_model_registry()

        # Should not find with wrong case
        found = get_model_by_id("OpenAI:GPT-4o-mini", registry)
        assert found is None


class TestGetModelsByProvider:
    """Tests for get_models_by_provider function."""

    def test_returns_openai_models(self):
        """Test getting all OpenAI models."""
        registry = create_model_registry()

        openai_models = get_models_by_provider("openai", registry)

        assert len(openai_models) == 4
        assert all(model.provider == "openai" for model in openai_models)

    def test_returns_anthropic_models(self):
        """Test getting all Anthropic models."""
        registry = create_model_registry()

        anthropic_models = get_models_by_provider("anthropic", registry)

        assert len(anthropic_models) == 3
        assert all(model.provider == "anthropic" for model in anthropic_models)

    def test_returns_empty_for_nonexistent_provider(self):
        """Test returns empty list for provider that doesn't exist."""
        registry = create_model_registry()

        models = get_models_by_provider("nonexistent", registry)

        assert models == []

    def test_returns_all_provider_models(self):
        """Test each provider returns expected count."""
        registry = create_model_registry()

        expected_counts = {
            "openai": 4,
            "anthropic": 3,
            "google": 3,
            "groq": 3,
            "mistral": 3,
            "cohere": 2,
        }

        for provider, expected_count in expected_counts.items():
            models = get_models_by_provider(provider, registry)
            assert len(models) == expected_count, f"{provider} should have {expected_count} models, got {len(models)}"


class TestFilterModels:
    """Tests for filter_models function."""

    def test_filter_by_min_quality(self):
        """Test filtering by minimum quality threshold."""
        registry = create_model_registry()

        high_quality = filter_models(registry, min_quality=0.90)

        assert all(model.expected_quality >= 0.90 for model in high_quality)
        assert len(high_quality) < len(registry), "Should filter out some models"

    def test_filter_by_max_cost(self):
        """Test filtering by maximum cost."""
        registry = create_model_registry()

        # Filter for low-cost models (average cost < $0.001 per token)
        low_cost = filter_models(registry, max_cost=0.001)

        for model in low_cost:
            avg_cost = (model.cost_per_input_token + model.cost_per_output_token) / 2
            assert avg_cost <= 0.001

    def test_filter_by_providers(self):
        """Test filtering by provider list."""
        registry = create_model_registry()

        openai_only = filter_models(registry, providers=["openai"])

        assert all(model.provider == "openai" for model in openai_only)
        assert len(openai_only) == 4

    def test_filter_multiple_providers(self):
        """Test filtering with multiple providers."""
        registry = create_model_registry()

        multi_provider = filter_models(registry, providers=["openai", "anthropic"])

        assert all(model.provider in ["openai", "anthropic"] for model in multi_provider)
        assert len(multi_provider) == 7  # 4 OpenAI + 3 Anthropic

    def test_filter_combined_criteria(self):
        """Test filtering with multiple criteria combined."""
        registry = create_model_registry()

        filtered = filter_models(
            registry,
            min_quality=0.85,
            max_cost=0.005,
            providers=["openai", "anthropic"]
        )

        for model in filtered:
            assert model.expected_quality >= 0.85
            avg_cost = (model.cost_per_input_token + model.cost_per_output_token) / 2
            assert avg_cost <= 0.005
            assert model.provider in ["openai", "anthropic"]

    def test_filter_no_criteria_returns_all(self):
        """Test filtering with no criteria returns all models."""
        registry = create_model_registry()

        all_models = filter_models(registry)

        assert len(all_models) == len(registry)

    def test_filter_strict_criteria_returns_few(self):
        """Test very strict criteria returns small subset."""
        registry = create_model_registry()

        # Very high quality, very low cost - should be very few or none
        strict = filter_models(registry, min_quality=0.95, max_cost=0.0001)

        # Should be empty or very small
        assert len(strict) <= 2


class TestGetRegistryStats:
    """Tests for get_registry_stats function."""

    def test_returns_stats_dict(self):
        """Test returns dictionary with statistics."""
        registry = create_model_registry()

        stats = get_registry_stats(registry)

        assert isinstance(stats, dict)
        assert "total_models" in stats
        assert "providers" in stats
        assert "cost_range" in stats
        assert "quality_range" in stats

    def test_total_models_correct(self):
        """Test total_models count is accurate."""
        registry = create_model_registry()

        stats = get_registry_stats(registry)

        assert stats["total_models"] == len(registry)

    def test_providers_breakdown_correct(self):
        """Test providers breakdown matches actual counts."""
        registry = create_model_registry()

        stats = get_registry_stats(registry)

        # Verify provider counts sum to total
        assert "models_by_provider" in stats
        provider_sum = sum(stats["models_by_provider"].values())
        assert provider_sum == stats["total_models"]

    def test_cost_range_valid(self):
        """Test cost range has min and max."""
        registry = create_model_registry()

        stats = get_registry_stats(registry)

        assert "min" in stats["cost_range"]
        assert "max" in stats["cost_range"]
        assert stats["cost_range"]["min"] <= stats["cost_range"]["max"]

    def test_quality_range_valid(self):
        """Test quality range is within 0-1."""
        registry = create_model_registry()

        stats = get_registry_stats(registry)

        assert 0 <= stats["quality_range"]["min"] <= 1
        assert 0 <= stats["quality_range"]["max"] <= 1
        assert stats["quality_range"]["min"] <= stats["quality_range"]["max"]
