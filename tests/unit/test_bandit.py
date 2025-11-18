"""Unit tests for ContextualBandit Thompson Sampling."""

import pytest

from conduit.engines.bandit import ContextualBandit
from conduit.core.models import ModelState, QueryFeatures


class TestContextualBandit:
    """Tests for ContextualBandit."""

    def test_initialization(self):
        """Test bandit initializes with uniform priors."""
        models = ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4"]
        bandit = ContextualBandit(models)

        assert len(bandit.model_states) == 3
        for model in models:
            state = bandit.model_states[model]
            assert state.alpha == 1.0
            assert state.beta == 1.0
            assert state.total_requests == 0

    def test_select_model_returns_valid_model(self):
        """Test model selection returns valid model from list."""
        models = ["gpt-4o-mini", "gpt-4o"]
        bandit = ContextualBandit(models)

        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=10,
            complexity_score=0.3,
            domain="general",
            domain_confidence=0.8,
        )

        selected = bandit.select_model(features)
        assert selected in models

    def test_select_model_with_subset(self):
        """Test model selection from subset of models."""
        models = ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4"]
        bandit = ContextualBandit(models)

        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=10,
            complexity_score=0.5,
            domain="code",
            domain_confidence=0.9,
        )

        # Select from subset
        subset = ["gpt-4o-mini", "gpt-4o"]
        selected = bandit.select_model(features, models=subset)
        assert selected in subset

    def test_select_model_empty_list_raises_error(self):
        """Test model selection with empty list raises ValueError."""
        bandit = ContextualBandit(["gpt-4o-mini"])

        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=10,
            complexity_score=0.5,
            domain="general",
            domain_confidence=0.8,
        )

        with pytest.raises(ValueError, match="No models available"):
            bandit.select_model(features, models=[])

    def test_update_success(self):
        """Test update with successful reward increases alpha."""
        bandit = ContextualBandit(["gpt-4o-mini"])

        initial_alpha = bandit.model_states["gpt-4o-mini"].alpha
        initial_beta = bandit.model_states["gpt-4o-mini"].beta

        # High reward (>= 0.7) should increment alpha
        bandit.update("gpt-4o-mini", reward=0.85, query_id="q123")

        assert bandit.model_states["gpt-4o-mini"].alpha == initial_alpha + 1.0
        assert bandit.model_states["gpt-4o-mini"].beta == initial_beta

    def test_update_failure(self):
        """Test update with low reward increases beta."""
        bandit = ContextualBandit(["gpt-4o-mini"])

        initial_alpha = bandit.model_states["gpt-4o-mini"].alpha
        initial_beta = bandit.model_states["gpt-4o-mini"].beta

        # Low reward (< 0.7) should increment beta
        bandit.update("gpt-4o-mini", reward=0.3, query_id="q123")

        assert bandit.model_states["gpt-4o-mini"].alpha == initial_alpha
        assert bandit.model_states["gpt-4o-mini"].beta == initial_beta + 1.0

    def test_update_threshold_boundary(self):
        """Test update at exact threshold counts as success."""
        bandit = ContextualBandit(["gpt-4o-mini"])

        initial_alpha = bandit.model_states["gpt-4o-mini"].alpha

        # Reward exactly at threshold (0.7) should increment alpha
        bandit.update("gpt-4o-mini", reward=0.7, query_id="q123")

        assert bandit.model_states["gpt-4o-mini"].alpha == initial_alpha + 1.0

    def test_update_custom_threshold(self):
        """Test update with custom success threshold."""
        bandit = ContextualBandit(["gpt-4o-mini"])

        initial_alpha = bandit.model_states["gpt-4o-mini"].alpha
        initial_beta = bandit.model_states["gpt-4o-mini"].beta

        # With threshold=0.9, reward=0.8 should increment beta
        bandit.update("gpt-4o-mini", reward=0.8, query_id="q123", success_threshold=0.9)

        assert bandit.model_states["gpt-4o-mini"].alpha == initial_alpha
        assert bandit.model_states["gpt-4o-mini"].beta == initial_beta + 1.0

    def test_update_increments_total_requests(self):
        """Test update increments total requests counter."""
        bandit = ContextualBandit(["gpt-4o-mini"])

        assert bandit.model_states["gpt-4o-mini"].total_requests == 0

        bandit.update("gpt-4o-mini", reward=0.85, query_id="q1")
        assert bandit.model_states["gpt-4o-mini"].total_requests == 1

        bandit.update("gpt-4o-mini", reward=0.3, query_id="q2")
        assert bandit.model_states["gpt-4o-mini"].total_requests == 2

    def test_update_unknown_model_raises_error(self):
        """Test update with unknown model raises ValueError."""
        bandit = ContextualBandit(["gpt-4o-mini"])

        with pytest.raises(ValueError, match="Unknown model"):
            bandit.update("unknown-model", reward=0.85, query_id="q123")

    def test_get_confidence_high_samples(self):
        """Test confidence increases with more samples."""
        bandit = ContextualBandit(["gpt-4o-mini"])

        # Initial confidence (high variance, low confidence)
        initial_confidence = bandit.get_confidence("gpt-4o-mini")

        # Add many successful samples
        for i in range(100):
            bandit.update("gpt-4o-mini", reward=0.9, query_id=f"q{i}")

        # Confidence should increase with more data
        final_confidence = bandit.get_confidence("gpt-4o-mini")
        assert final_confidence > initial_confidence

    def test_get_confidence_unknown_model(self):
        """Test get_confidence returns 0.0 for unknown model."""
        bandit = ContextualBandit(["gpt-4o-mini"])

        confidence = bandit.get_confidence("unknown-model")
        assert confidence == 0.0

    def test_get_model_state(self):
        """Test get_model_state returns current state."""
        bandit = ContextualBandit(["gpt-4o-mini"])

        bandit.update("gpt-4o-mini", reward=0.85, query_id="q123")

        state = bandit.get_model_state("gpt-4o-mini")
        assert isinstance(state, ModelState)
        assert state.model_id == "gpt-4o-mini"
        assert state.alpha == 2.0  # 1.0 + 1.0
        assert state.beta == 1.0
        assert state.total_requests == 1

    def test_get_model_state_unknown_model_raises_error(self):
        """Test get_model_state with unknown model raises ValueError."""
        bandit = ContextualBandit(["gpt-4o-mini"])

        with pytest.raises(ValueError, match="Unknown model"):
            bandit.get_model_state("unknown-model")

    def test_load_states(self):
        """Test loading model states from database."""
        bandit = ContextualBandit(["gpt-4o-mini", "gpt-4o"])

        # Create states with non-default values
        states = {
            "gpt-4o-mini": ModelState(
                model_id="gpt-4o-mini", alpha=10.0, beta=5.0, total_requests=100
            ),
            "gpt-4o": ModelState(model_id="gpt-4o", alpha=20.0, beta=3.0, total_requests=200),
        }

        bandit.load_states(states)

        assert bandit.model_states["gpt-4o-mini"].alpha == 10.0
        assert bandit.model_states["gpt-4o-mini"].beta == 5.0
        assert bandit.model_states["gpt-4o"].alpha == 20.0
        assert bandit.model_states["gpt-4o"].beta == 3.0

    def test_predict_reward_simple_query(self):
        """Test reward prediction for simple queries."""
        bandit = ContextualBandit(["gpt-4o-mini", "claude-opus-4"])

        # Simple query should prefer cheap model
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=5,
            complexity_score=0.1,  # Very simple
            domain="general",
            domain_confidence=0.8,
        )

        mini_reward = bandit._predict_reward("gpt-4o-mini", features)
        opus_reward = bandit._predict_reward("claude-opus-4", features)

        assert mini_reward > opus_reward

    def test_predict_reward_complex_query(self):
        """Test reward prediction for complex queries."""
        bandit = ContextualBandit(["gpt-4o-mini", "claude-opus-4"])

        # Complex query should prefer premium model
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=100,
            complexity_score=0.9,  # Very complex
            domain="code",
            domain_confidence=0.9,
        )

        mini_reward = bandit._predict_reward("gpt-4o-mini", features)
        opus_reward = bandit._predict_reward("claude-opus-4", features)

        assert opus_reward > mini_reward
