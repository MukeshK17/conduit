"""Tests for optimistic locking in PostgresStateStore.

Tests cover:
- Version conflict detection
- Retry with exponential backoff
- Concurrent writes handling
- StateVersionConflictError after max retries
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conduit.core.postgres_state_store import (
    BASE_DELAY_MS,
    MAX_DELAY_MS,
    MAX_RETRIES,
    PostgresStateStore,
    StateVersionConflictError,
)
from conduit.core.state_store import BanditState, HybridRouterState


@pytest.fixture
def mock_pool():
    """Create mock database pool."""
    pool = MagicMock()
    pool.acquire = MagicMock()
    return pool


@pytest.fixture
def state_store(mock_pool):
    """Create PostgresStateStore with mock pool."""
    return PostgresStateStore(mock_pool)


@pytest.fixture
def sample_bandit_state():
    """Create sample BanditState for testing."""
    return BanditState(
        algorithm="thompson_sampling",
        arm_ids=["gpt-4o-mini", "gpt-4o"],
        arm_pulls={"gpt-4o-mini": 5, "gpt-4o": 5},
        arm_successes={"gpt-4o-mini": 4, "gpt-4o": 3},
        total_queries=10,
        alpha_params={"gpt-4o-mini": 2.0, "gpt-4o": 1.5},
        beta_params={"gpt-4o-mini": 1.0, "gpt-4o": 1.5},
    )


@pytest.fixture
def sample_hybrid_state():
    """Create sample HybridRouterState for testing."""
    return HybridRouterState(
        query_count=10,
        phase1_state=BanditState(
            algorithm="thompson_sampling",
            arm_ids=["gpt-4o-mini", "gpt-4o"],
            arm_pulls={"gpt-4o-mini": 5, "gpt-4o": 5},
            total_queries=10,
        ),
    )


class TestBackoffCalculation:
    """Tests for exponential backoff delay calculation."""

    def test_backoff_increases_exponentially(self, state_store):
        """Test that backoff delay increases with each attempt."""
        delays = [state_store._calculate_backoff_delay(i) for i in range(5)]
        # Each delay should be roughly 2x the previous (with jitter)
        # BASE_DELAY = 50ms, so: 50, 100, 200, 400, 500 (capped)
        for i in range(1, len(delays)):
            # Allow for jitter variance
            assert delays[i] >= delays[i - 1] * 0.8

    def test_backoff_respects_max_delay(self, state_store):
        """Test that backoff is capped at MAX_DELAY_MS."""
        delay = state_store._calculate_backoff_delay(10)  # Large attempt number
        # Maximum delay should be MAX_DELAY_MS + 50% jitter
        assert delay <= (MAX_DELAY_MS * 1.5) / 1000.0

    def test_backoff_has_jitter(self, state_store):
        """Test that backoff includes randomized jitter."""
        delays = [state_store._calculate_backoff_delay(0) for _ in range(10)]
        # With jitter, not all delays should be identical
        unique_delays = {round(d, 6) for d in delays}
        assert len(unique_delays) > 1


class TestOptimisticLockingBanditState:
    """Tests for optimistic locking in save_bandit_state."""

    @pytest.mark.asyncio
    async def test_insert_new_state_succeeds(
        self, state_store, sample_bandit_state, mock_pool
    ):
        """Test that inserting new state succeeds on first attempt."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                None,  # _get_current_version returns None (no existing record)
                {"version": 1},  # INSERT succeeds
            ]
        )
        mock_conn.execute = AsyncMock()

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        await state_store.save_bandit_state("router-1", "thompson", sample_bandit_state)

        # Should have called fetchrow twice: version check + insert
        assert mock_conn.fetchrow.call_count == 2

    @pytest.mark.asyncio
    async def test_update_existing_state_succeeds(
        self, state_store, sample_bandit_state, mock_pool
    ):
        """Test that updating existing state succeeds when version matches."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                {"version": 5},  # _get_current_version returns 5
                {"version": 6},  # UPDATE succeeds, new version is 6
            ]
        )

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        await state_store.save_bandit_state("router-1", "thompson", sample_bandit_state)

        # Should have called fetchrow twice: version check + update
        assert mock_conn.fetchrow.call_count == 2

    @pytest.mark.asyncio
    async def test_version_conflict_triggers_retry(
        self, state_store, sample_bandit_state, mock_pool
    ):
        """Test that version conflict triggers retry with backoff."""
        mock_conn = AsyncMock()
        # First attempt: conflict (version changed)
        # Second attempt: success
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                {"version": 5},  # First version check
                None,  # UPDATE fails (version mismatch)
                {"version": 6},  # Second version check (after retry)
                {"version": 7},  # UPDATE succeeds
            ]
        )

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await state_store.save_bandit_state(
                "router-1", "thompson", sample_bandit_state
            )
            # Should have slept once for backoff
            mock_sleep.assert_called_once()

        assert state_store.conflict_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_raises_conflict_error(
        self, state_store, sample_bandit_state, mock_pool
    ):
        """Test that StateVersionConflictError is raised after max retries."""
        mock_conn = AsyncMock()
        # All attempts fail with version conflict
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                {"version": i} if i % 2 == 0 else None
                for i in range(20)  # Enough for MAX_RETRIES + 1 attempts
            ]
        )

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(StateVersionConflictError) as exc_info,
        ):
            await state_store.save_bandit_state(
                "router-1", "thompson", sample_bandit_state
            )

        assert f"{MAX_RETRIES} retries" in str(exc_info.value)
        assert state_store.conflict_count == MAX_RETRIES + 1

    @pytest.mark.asyncio
    async def test_insert_conflict_retries_as_update(
        self, state_store, sample_bandit_state, mock_pool
    ):
        """Test that insert conflict (race) retries as update."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                None,  # _get_current_version returns None (looks new)
                None,  # INSERT fails (another process inserted first)
                {"version": 1},  # _get_current_version now returns 1
                {"version": 2},  # UPDATE succeeds
            ]
        )

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        await state_store.save_bandit_state("router-1", "thompson", sample_bandit_state)

        # Should have made 4 fetchrow calls
        assert mock_conn.fetchrow.call_count == 4


class TestOptimisticLockingHybridState:
    """Tests for optimistic locking in save_hybrid_router_state."""

    @pytest.mark.asyncio
    async def test_hybrid_insert_succeeds(
        self, state_store, sample_hybrid_state, mock_pool
    ):
        """Test that inserting hybrid router state succeeds."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                None,  # No existing record
                {"version": 1},  # INSERT succeeds
            ]
        )

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        await state_store.save_hybrid_router_state("router-1", sample_hybrid_state)

        assert mock_conn.fetchrow.call_count == 2

    @pytest.mark.asyncio
    async def test_hybrid_update_with_version_check(
        self, state_store, sample_hybrid_state, mock_pool
    ):
        """Test that hybrid state update checks version."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                {"version": 10},  # Existing record at version 10
                {"version": 11},  # UPDATE succeeds
            ]
        )

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        await state_store.save_hybrid_router_state("router-1", sample_hybrid_state)

        # Verify version was passed to update query
        update_call = mock_conn.fetchrow.call_args_list[1]
        assert 10 in update_call[0]  # Version 10 should be in the query args

    @pytest.mark.asyncio
    async def test_hybrid_conflict_retries(
        self, state_store, sample_hybrid_state, mock_pool
    ):
        """Test that hybrid state version conflict triggers retry."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                {"version": 5},  # First check
                None,  # Conflict
                {"version": 6},  # Second check
                {"version": 7},  # Success
            ]
        )

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await state_store.save_hybrid_router_state("router-1", sample_hybrid_state)

        assert state_store.conflict_count == 1


class TestConflictMetrics:
    """Tests for conflict counting and monitoring."""

    def test_conflict_count_initializes_to_zero(self, state_store):
        """Test that conflict count starts at zero."""
        assert state_store.conflict_count == 0

    @pytest.mark.asyncio
    async def test_conflict_count_increments(
        self, state_store, sample_bandit_state, mock_pool
    ):
        """Test that conflict count increments on each conflict."""
        mock_conn = AsyncMock()
        # Two conflicts before success
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                {"version": 1},
                None,  # Conflict 1
                {"version": 2},
                None,  # Conflict 2
                {"version": 3},
                {"version": 4},  # Success
            ]
        )

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await state_store.save_bandit_state("router-1", "test", sample_bandit_state)

        assert state_store.conflict_count == 2

    @pytest.mark.asyncio
    async def test_conflict_count_accumulates_across_calls(
        self, state_store, sample_bandit_state, mock_pool
    ):
        """Test that conflict count accumulates across multiple save calls."""
        mock_conn = AsyncMock()

        # First save: 1 conflict
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                {"version": 1},
                None,  # Conflict
                {"version": 2},
                {"version": 3},  # Success
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await state_store.save_bandit_state(
                "router-1", "test1", sample_bandit_state
            )

        assert state_store.conflict_count == 1

        # Second save: 1 more conflict
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                {"version": 1},
                None,  # Conflict
                {"version": 2},
                {"version": 3},  # Success
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await state_store.save_bandit_state(
                "router-1", "test2", sample_bandit_state
            )

        assert state_store.conflict_count == 2
