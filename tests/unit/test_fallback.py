"""Tests for fallback chain functionality.

Tests cover:
- RoutingDecision fallback_chain generation
- execute_with_fallback() behavior
- Feedback attribution for fallback scenarios
- AllModelsFailed exception handling
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from conduit.core.exceptions import ExecutionError
from conduit.core.models import QueryFeatures, Response, RoutingDecision
from conduit.engines.executor import AllModelsFailedError, ExecutionResult, ModelExecutor


class TestOutput(BaseModel):
    """Test output model for executor tests."""

    answer: str


@pytest.fixture
def mock_features() -> QueryFeatures:
    """Create mock query features for testing."""
    return QueryFeatures(
        embedding=[0.1] * 384,
        token_count=50,
        complexity_score=0.5,
        query_text="test query",
    )


@pytest.fixture
def mock_decision(mock_features: QueryFeatures) -> RoutingDecision:
    """Create mock routing decision with fallback chain."""
    return RoutingDecision(
        id="test-decision-id",
        query_id="test-query-id",
        selected_model="gpt-4o",
        fallback_chain=["gpt-4o-mini", "claude-3-haiku", "gemini-flash"],
        confidence=0.85,
        features=mock_features,
        reasoning="Test routing decision",
    )


@pytest.fixture
def executor() -> ModelExecutor:
    """Create model executor for testing."""
    return ModelExecutor()


class TestRoutingDecisionFallbackChain:
    """Tests for fallback_chain field in RoutingDecision."""

    def test_fallback_chain_default_empty(self, mock_features: QueryFeatures) -> None:
        """Test that fallback_chain defaults to empty list."""
        decision = RoutingDecision(
            query_id="test-id",
            selected_model="gpt-4o",
            confidence=0.9,
            features=mock_features,
            reasoning="test",
        )
        assert decision.fallback_chain == []

    def test_fallback_chain_with_models(self, mock_features: QueryFeatures) -> None:
        """Test that fallback_chain stores model list correctly."""
        fallbacks = ["gpt-4o-mini", "claude-3-haiku", "gemini-flash"]
        decision = RoutingDecision(
            query_id="test-id",
            selected_model="gpt-4o",
            fallback_chain=fallbacks,
            confidence=0.9,
            features=mock_features,
            reasoning="test",
        )
        assert decision.fallback_chain == fallbacks
        assert len(decision.fallback_chain) == 3

    def test_fallback_chain_preserves_order(self, mock_features: QueryFeatures) -> None:
        """Test that fallback_chain preserves model ordering."""
        fallbacks = ["model-a", "model-b", "model-c"]
        decision = RoutingDecision(
            query_id="test-id",
            selected_model="primary",
            fallback_chain=fallbacks,
            confidence=0.9,
            features=mock_features,
            reasoning="test",
        )
        assert decision.fallback_chain[0] == "model-a"
        assert decision.fallback_chain[1] == "model-b"
        assert decision.fallback_chain[2] == "model-c"


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_basic_result(self) -> None:
        """Test basic ExecutionResult creation."""
        response = Response(
            id="resp-id",
            query_id="query-id",
            model="gpt-4o",
            text='{"answer": "test"}',
            cost=0.01,
            latency=0.5,
            tokens=100,
        )
        result = ExecutionResult(
            response=response,
            model_used="gpt-4o",
        )
        assert result.response == response
        assert result.model_used == "gpt-4o"
        assert result.was_fallback is False
        assert result.original_model == ""
        assert result.failed_models == []

    def test_fallback_result(self) -> None:
        """Test ExecutionResult with fallback metadata."""
        response = Response(
            id="resp-id",
            query_id="query-id",
            model="gpt-4o-mini",
            text='{"answer": "test"}',
            cost=0.001,
            latency=0.3,
            tokens=50,
        )
        result = ExecutionResult(
            response=response,
            model_used="gpt-4o-mini",
            was_fallback=True,
            original_model="gpt-4o",
            failed_models=["gpt-4o"],
        )
        assert result.was_fallback is True
        assert result.original_model == "gpt-4o"
        assert result.failed_models == ["gpt-4o"]

    def test_multiple_failures_result(self) -> None:
        """Test ExecutionResult with multiple failed models."""
        response = Response(
            id="resp-id",
            query_id="query-id",
            model="gemini-flash",
            text='{"answer": "test"}',
            cost=0.0001,
            latency=0.2,
            tokens=30,
        )
        result = ExecutionResult(
            response=response,
            model_used="gemini-flash",
            was_fallback=True,
            original_model="gpt-4o",
            failed_models=["gpt-4o", "gpt-4o-mini", "claude-3-haiku"],
        )
        assert len(result.failed_models) == 3
        assert "gpt-4o" in result.failed_models


class TestAllModelsFailedError:
    """Tests for AllModelsFailedError exception."""

    def test_exception_creation(self) -> None:
        """Test AllModelsFailedError exception creation."""
        errors = [
            ("gpt-4o", ExecutionError("Rate limited")),
            ("gpt-4o-mini", ExecutionError("Timeout")),
        ]
        exc = AllModelsFailedError("All models failed", errors=errors)
        assert "All models failed" in str(exc)
        assert exc.errors == errors
        assert len(exc.errors) == 2

    def test_exception_is_execution_error(self) -> None:
        """Test that AllModelsFailedError inherits from ExecutionError."""
        exc = AllModelsFailedError("test", errors=[])
        assert isinstance(exc, ExecutionError)


class TestExecuteWithFallback:
    """Tests for execute_with_fallback() method."""

    @pytest.mark.asyncio
    async def test_primary_success_no_fallback(
        self,
        executor: ModelExecutor,
        mock_decision: RoutingDecision,
    ) -> None:
        """Test successful execution without needing fallback."""
        mock_response = Response(
            id="resp-id",
            query_id=mock_decision.query_id,
            model="gpt-4o",
            text='{"answer": "success"}',
            cost=0.01,
            latency=0.5,
            tokens=100,
        )

        with patch.object(executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_response

            result = await executor.execute_with_fallback(
                decision=mock_decision,
                prompt="test prompt",
                result_type=TestOutput,
            )

            assert result.model_used == "gpt-4o"
            assert result.was_fallback is False
            assert result.failed_models == []
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_single_failure_fallback_success(
        self,
        executor: ModelExecutor,
        mock_decision: RoutingDecision,
    ) -> None:
        """Test fallback to second model after primary fails."""
        mock_response = Response(
            id="resp-id",
            query_id=mock_decision.query_id,
            model="gpt-4o-mini",
            text='{"answer": "fallback success"}',
            cost=0.001,
            latency=0.3,
            tokens=50,
        )

        call_count = 0

        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ExecutionError("Primary model failed")
            return mock_response

        with patch.object(executor, "execute", side_effect=mock_execute):
            result = await executor.execute_with_fallback(
                decision=mock_decision,
                prompt="test prompt",
                result_type=TestOutput,
            )

            assert result.model_used == "gpt-4o-mini"
            assert result.was_fallback is True
            assert result.original_model == "gpt-4o"
            assert result.failed_models == ["gpt-4o"]

    @pytest.mark.asyncio
    async def test_multiple_failures_fallback_success(
        self,
        executor: ModelExecutor,
        mock_decision: RoutingDecision,
    ) -> None:
        """Test fallback after multiple model failures."""
        mock_response = Response(
            id="resp-id",
            query_id=mock_decision.query_id,
            model="claude-3-haiku",
            text='{"answer": "third try success"}',
            cost=0.0005,
            latency=0.25,
            tokens=40,
        )

        call_count = 0

        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ExecutionError(f"Model {call_count} failed")
            return mock_response

        with patch.object(executor, "execute", side_effect=mock_execute):
            result = await executor.execute_with_fallback(
                decision=mock_decision,
                prompt="test prompt",
                result_type=TestOutput,
            )

            assert result.model_used == "claude-3-haiku"
            assert result.was_fallback is True
            assert len(result.failed_models) == 2
            assert "gpt-4o" in result.failed_models
            assert "gpt-4o-mini" in result.failed_models

    @pytest.mark.asyncio
    async def test_all_models_fail(
        self,
        executor: ModelExecutor,
        mock_decision: RoutingDecision,
    ) -> None:
        """Test AllModelsFailedError raised when all models fail."""
        async def mock_execute(*args, **kwargs):
            raise ExecutionError("Model unavailable")

        with patch.object(executor, "execute", side_effect=mock_execute):
            with pytest.raises(AllModelsFailedError) as exc_info:
                await executor.execute_with_fallback(
                    decision=mock_decision,
                    prompt="test prompt",
                    result_type=TestOutput,
                )

            assert len(exc_info.value.errors) == 4  # primary + 3 fallbacks
            assert "gpt-4o" in exc_info.value.errors[0][0]

    @pytest.mark.asyncio
    async def test_max_fallbacks_respected(
        self,
        executor: ModelExecutor,
        mock_features: QueryFeatures,
    ) -> None:
        """Test that max_fallbacks parameter limits retry attempts."""
        # Decision with many fallbacks
        decision = RoutingDecision(
            query_id="test-id",
            selected_model="primary",
            fallback_chain=["fb1", "fb2", "fb3", "fb4", "fb5"],
            confidence=0.9,
            features=mock_features,
            reasoning="test",
        )

        call_count = 0

        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ExecutionError("Always fails")

        with patch.object(executor, "execute", side_effect=mock_execute):
            with pytest.raises(AllModelsFailedError) as exc_info:
                await executor.execute_with_fallback(
                    decision=decision,
                    prompt="test",
                    result_type=TestOutput,
                    max_fallbacks=2,  # Only try 2 fallbacks
                )

            # Should try: primary + 2 fallbacks = 3 total
            assert call_count == 3
            assert len(exc_info.value.errors) == 3

    @pytest.mark.asyncio
    async def test_timeout_error_triggers_fallback(
        self,
        executor: ModelExecutor,
        mock_decision: RoutingDecision,
    ) -> None:
        """Test that timeout errors also trigger fallback."""
        mock_response = Response(
            id="resp-id",
            query_id=mock_decision.query_id,
            model="gpt-4o-mini",
            text='{"answer": "after timeout"}',
            cost=0.001,
            latency=0.3,
            tokens=50,
        )

        call_count = 0

        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError("Request timed out")
            return mock_response

        with patch.object(executor, "execute", side_effect=mock_execute):
            result = await executor.execute_with_fallback(
                decision=mock_decision,
                prompt="test prompt",
                result_type=TestOutput,
            )

            assert result.was_fallback is True
            assert result.model_used == "gpt-4o-mini"


class TestFeedbackAttribution:
    """Tests for Router.update_with_fallback_attribution()."""

    @pytest.mark.asyncio
    async def test_no_fallback_single_update(self) -> None:
        """Test that non-fallback case only updates once."""
        from conduit.engines.router import Router

        response = Response(
            id="resp-id",
            query_id="query-id",
            model="gpt-4o",
            text='{"answer": "test"}',
            cost=0.01,
            latency=0.5,
            tokens=100,
        )
        result = ExecutionResult(
            response=response,
            model_used="gpt-4o",
            was_fallback=False,
            original_model="gpt-4o",
            failed_models=[],
        )
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.5,
            query_text="test",
        )

        with patch.object(Router, "__init__", return_value=None):
            router = Router.__new__(Router)
            router.hybrid_router = MagicMock()
            router.auto_persist = False

            update_calls = []

            async def mock_update(model_id, cost, quality_score, latency, features):
                update_calls.append(
                    {
                        "model_id": model_id,
                        "cost": cost,
                        "quality_score": quality_score,
                        "latency": latency,
                    }
                )

            router.update = mock_update

            await router.update_with_fallback_attribution(
                execution_result=result,
                quality_score=0.95,
                features=features,
            )

            assert len(update_calls) == 1
            assert update_calls[0]["model_id"] == "gpt-4o"
            assert update_calls[0]["quality_score"] == 0.95

    @pytest.mark.asyncio
    async def test_fallback_penalizes_failed_models(self) -> None:
        """Test that failed models receive quality=0.0 penalty."""
        from conduit.engines.router import Router

        response = Response(
            id="resp-id",
            query_id="query-id",
            model="gpt-4o-mini",
            text='{"answer": "test"}',
            cost=0.001,
            latency=0.3,
            tokens=50,
        )
        result = ExecutionResult(
            response=response,
            model_used="gpt-4o-mini",
            was_fallback=True,
            original_model="gpt-4o",
            failed_models=["gpt-4o"],
        )
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.5,
            query_text="test",
        )

        with patch.object(Router, "__init__", return_value=None):
            router = Router.__new__(Router)
            router.hybrid_router = MagicMock()
            router.auto_persist = False

            update_calls = []

            async def mock_update(model_id, cost, quality_score, latency, features):
                update_calls.append(
                    {
                        "model_id": model_id,
                        "cost": cost,
                        "quality_score": quality_score,
                        "latency": latency,
                    }
                )

            router.update = mock_update

            await router.update_with_fallback_attribution(
                execution_result=result,
                quality_score=0.95,
                features=features,
            )

            # Should have 2 calls: 1 for failed model, 1 for successful
            assert len(update_calls) == 2

            # Failed model should be penalized
            failed_update = update_calls[0]
            assert failed_update["model_id"] == "gpt-4o"
            assert failed_update["quality_score"] == 0.0
            assert failed_update["cost"] == 0.0

            # Successful model should be rewarded
            success_update = update_calls[1]
            assert success_update["model_id"] == "gpt-4o-mini"
            assert success_update["quality_score"] == 0.95
            assert success_update["cost"] == 0.001

    @pytest.mark.asyncio
    async def test_multiple_failures_all_penalized(self) -> None:
        """Test that all failed models receive penalties."""
        from conduit.engines.router import Router

        response = Response(
            id="resp-id",
            query_id="query-id",
            model="gemini-flash",
            text='{"answer": "test"}',
            cost=0.0001,
            latency=0.2,
            tokens=30,
        )
        result = ExecutionResult(
            response=response,
            model_used="gemini-flash",
            was_fallback=True,
            original_model="gpt-4o",
            failed_models=["gpt-4o", "gpt-4o-mini", "claude-3-haiku"],
        )
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.5,
            query_text="test",
        )

        with patch.object(Router, "__init__", return_value=None):
            router = Router.__new__(Router)
            router.hybrid_router = MagicMock()
            router.auto_persist = False

            update_calls = []

            async def mock_update(model_id, cost, quality_score, latency, features):
                update_calls.append(
                    {
                        "model_id": model_id,
                        "quality_score": quality_score,
                    }
                )

            router.update = mock_update

            await router.update_with_fallback_attribution(
                execution_result=result,
                quality_score=0.90,
                features=features,
            )

            # 3 failed + 1 success = 4 calls
            assert len(update_calls) == 4

            # First 3 should be penalties
            for i in range(3):
                assert update_calls[i]["quality_score"] == 0.0

            # Last should be reward
            assert update_calls[3]["model_id"] == "gemini-flash"
            assert update_calls[3]["quality_score"] == 0.90
