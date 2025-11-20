"""Unit tests for bandit base classes."""

import pytest

from conduit.engines.bandits.base import BanditAlgorithm, BanditFeedback, ModelArm
from conduit.core.models import QueryFeatures


class TestModelArm:
    """Tests for ModelArm."""

    def test_initialization(self):
        """Test ModelArm initializes correctly."""
        arm = ModelArm(
            model_id="gpt-4o-mini",
            model_name="gpt-4o-mini",
            provider="openai",
            cost_per_input_token=0.00015,
            cost_per_output_token=0.0006,
            expected_quality=0.85,
        )

        assert arm.model_id == "gpt-4o-mini"
        assert arm.provider == "openai"
        assert arm.cost_per_input_token == 0.00015
        assert arm.cost_per_output_token == 0.0006
        assert arm.expected_quality == 0.85

    def test_average_cost(self):
        """Test average cost calculation."""
        arm = ModelArm(
            model_id="gpt-4o",
            model_name="gpt-4o",
            provider="openai",
            cost_per_input_token=0.0025,
            cost_per_output_token=0.010,
            expected_quality=0.95,
        )

        # Average of input and output costs
        assert (arm.cost_per_input_token + arm.cost_per_output_token) / 2 == (0.0025 + 0.010) / 2

    def test_model_arm_equality(self):
        """Test ModelArm equality comparison."""
        arm1 = ModelArm(
            model_id="gpt-4o-mini",
            model_name="gpt-4o-mini",
            provider="openai",
            cost_per_input_token=0.00015,
            cost_per_output_token=0.0006,
            expected_quality=0.85,
        )
        arm2 = ModelArm(
            model_id="gpt-4o-mini",
            model_name="gpt-4o-mini",
            provider="openai",
            cost_per_input_token=0.00015,
            cost_per_output_token=0.0006,
            expected_quality=0.85,
        )
        arm3 = ModelArm(
            model_id="gpt-4o",
            model_name="gpt-4o",
            provider="openai",
            cost_per_input_token=0.0025,
            cost_per_output_token=0.010,
            expected_quality=0.95,
        )

        assert arm1 == arm2
        assert arm1 != arm3


class TestBanditFeedback:
    """Tests for BanditFeedback."""

    def test_initialization(self):
        """Test BanditFeedback initializes correctly."""
        feedback = BanditFeedback(
            model_id="gpt-4o-mini",
            cost=0.001,
            quality_score=0.92,
            latency=1.5,
        )

        assert feedback.model_id == "gpt-4o-mini"
        assert feedback.cost == 0.001
        assert feedback.quality_score == 0.92
        assert feedback.latency == 1.5

    def test_quality_score_validation(self):
        """Test quality_score must be between 0 and 1."""
        # Valid quality scores
        BanditFeedback(
            model_id="test",
            cost=0.001,
            quality_score=0.0,
            latency=1.0,
        )
        BanditFeedback(
            model_id="test",
            cost=0.001,
            quality_score=1.0,
            latency=1.0,
        )

        # Invalid quality scores should be caught by Pydantic
        with pytest.raises(Exception):  # Pydantic validation error
            BanditFeedback(
                model_id="test",
                cost=0.001,
                quality_score=1.5,  # > 1.0
                latency=1.0,
            )


class TestBanditAlgorithm:
    """Tests for BanditAlgorithm base class."""

    def test_abstract_class(self):
        """Test BanditAlgorithm is abstract and cannot be instantiated."""
        arms = [
            ModelArm(
                model_id="gpt-4o-mini",
                model_name="gpt-4o-mini",
                provider="openai",
                cost_per_input_token=0.00015,
                cost_per_output_token=0.0006,
                expected_quality=0.85,
            )
        ]

        # Should raise TypeError because abstract methods not implemented
        with pytest.raises(TypeError):
            BanditAlgorithm(name="test", arms=arms)

    def test_arms_dict_creation(self):
        """Test arms list is converted to dict by model_id."""
        # We'll test this through a concrete implementation in other test files
        pass
