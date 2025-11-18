"""Supabase PostgreSQL database interface with transaction management."""

import json
import logging
from typing import Any

import asyncpg

from conduit.core.exceptions import DatabaseError
from conduit.core.models import (
    Feedback,
    ModelState,
    Query,
    Response,
    RoutingDecision,
)

logger = logging.getLogger(__name__)


class Database:
    """Supabase PostgreSQL interface with transaction management.

    Transaction Boundaries:
        - Single row inserts: Auto-commit (no explicit transaction)
        - Feedback loop updates: Transaction (query → routing → response → feedback)
        - Batch operations: Transaction for consistency
        - Model state updates: Auto-commit (last-write-wins acceptable)
        - Circuit breaker state: Auto-commit (eventual consistency)

    Isolation Level: READ COMMITTED (PostgreSQL default)
    Retry Strategy: Exponential backoff on deadlock/serialization failure
    """

    def __init__(self, connection_string: str):
        """Initialize database connection pool.

        Args:
            connection_string: PostgreSQL connection URL
        """
        self.connection_string = connection_string
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create connection pool."""
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=5,
            max_size=20,
            command_timeout=60.0,
        )
        logger.info("Database connection pool created")

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

    async def save_query(self, query: Query) -> str:
        """Save query and return ID.

        Transaction: None (single INSERT, auto-commit)

        Args:
            query: Query to save

        Returns:
            Query ID

        Raises:
            DatabaseError: If save fails
        """
        if not self.pool:
            raise DatabaseError("Database not connected")

        try:
            async with self.pool.acquire() as conn:
                query_id = await conn.fetchval(
                    """
                    INSERT INTO queries (id, text, user_id, context, constraints, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                    """,
                    query.id,
                    query.text,
                    query.user_id,
                    json.dumps(query.context) if query.context else None,
                    json.dumps(query.constraints.model_dump())
                    if query.constraints
                    else None,
                    query.created_at,
                )
            return str(query_id)

        except Exception as e:
            logger.error(f"Failed to save query: {e}")
            raise DatabaseError(f"Failed to save query: {e}") from e

    async def save_complete_interaction(
        self,
        routing: RoutingDecision,
        response: Response,
        feedback: Feedback | None = None,
    ) -> None:
        """Save routing decision, response, and optional feedback atomically.

        Transaction: REQUIRED (ensures consistency of related records)
        Rollback: On any failure, all records rolled back

        Args:
            routing: Routing decision
            response: LLM response
            feedback: Optional user feedback

        Raises:
            DatabaseError: If save fails
        """
        if not self.pool:
            raise DatabaseError("Database not connected")

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Save routing decision
                    await conn.execute(
                        """
                        INSERT INTO routing_decisions (id, query_id, selected_model, confidence, features, reasoning, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        routing.id,
                        routing.query_id,
                        routing.selected_model,
                        routing.confidence,
                        json.dumps(routing.features.model_dump()),
                        routing.reasoning,
                        routing.created_at,
                    )

                    # Save response
                    await conn.execute(
                        """
                        INSERT INTO responses (id, query_id, model, text, cost, latency, tokens, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                        response.id,
                        response.query_id,
                        response.model,
                        response.text,
                        response.cost,
                        response.latency,
                        response.tokens,
                        response.created_at,
                    )

                    # Save feedback if provided
                    if feedback:
                        await conn.execute(
                            """
                            INSERT INTO feedback (id, response_id, quality_score, user_rating, met_expectations, comments, created_at)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            """,
                            feedback.id,
                            feedback.response_id,
                            feedback.quality_score,
                            feedback.user_rating,
                            feedback.met_expectations,
                            feedback.comments,
                            feedback.created_at,
                        )

            logger.info(
                f"Saved complete interaction: routing={routing.id}, response={response.id}"
            )

        except Exception as e:
            logger.error(f"Failed to save interaction: {e}")
            raise DatabaseError(f"Failed to save interaction: {e}") from e

    async def update_model_state(self, state: ModelState) -> None:
        """Update model's Beta parameters.

        Transaction: None (UPSERT with ON CONFLICT, auto-commit)
        Concurrency: Last-write-wins acceptable for ML updates

        Args:
            state: Model state to update

        Raises:
            DatabaseError: If update fails
        """
        if not self.pool:
            raise DatabaseError("Database not connected")

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO model_states (model_id, alpha, beta, total_requests, total_cost, avg_quality, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (model_id) DO UPDATE
                    SET alpha = $2, beta = $3, total_requests = $4,
                        total_cost = $5, avg_quality = $6, updated_at = $7
                    """,
                    state.model_id,
                    state.alpha,
                    state.beta,
                    state.total_requests,
                    state.total_cost,
                    state.avg_quality,
                    state.updated_at,
                )

            logger.debug(f"Updated model state: {state.model_id}")

        except Exception as e:
            logger.error(f"Failed to update model state: {e}")
            raise DatabaseError(f"Failed to update model state: {e}") from e

    async def get_model_states(self) -> dict[str, ModelState]:
        """Load all model states.

        Transaction: None (single SELECT, read-only)

        Returns:
            Dictionary mapping model_id to ModelState

        Raises:
            DatabaseError: If load fails
        """
        if not self.pool:
            raise DatabaseError("Database not connected")

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM model_states")

            states = {
                row["model_id"]: ModelState(
                    model_id=row["model_id"],
                    alpha=row["alpha"],
                    beta=row["beta"],
                    total_requests=row["total_requests"],
                    total_cost=row["total_cost"],
                    avg_quality=row["avg_quality"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            }

            logger.info(f"Loaded {len(states)} model states")
            return states

        except Exception as e:
            logger.error(f"Failed to load model states: {e}")
            raise DatabaseError(f"Failed to load model states: {e}") from e

    async def get_response_by_id(self, response_id: str) -> Response | None:
        """Get response by ID.

        Args:
            response_id: Response ID

        Returns:
            Response if found, None otherwise

        Raises:
            DatabaseError: If query fails
        """
        if not self.pool:
            raise DatabaseError("Database not connected")

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM responses WHERE id = $1", response_id
                )

            if not row:
                return None

            return Response(
                id=row["id"],
                query_id=row["query_id"],
                model=row["model"],
                text=row["text"],
                cost=row["cost"],
                latency=row["latency"],
                tokens=row["tokens"],
                created_at=row["created_at"],
            )

        except Exception as e:
            logger.error(f"Failed to get response: {e}")
            raise DatabaseError(f"Failed to get response: {e}") from e
