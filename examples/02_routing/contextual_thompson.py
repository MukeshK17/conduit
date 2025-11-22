"""Contextual Thompson Sampling routing example.

This example demonstrates using Contextual Thompson Sampling (Bayesian linear regression)
for intelligent model selection. This algorithm combines Thompson Sampling's natural
exploration with contextual features (query embeddings, complexity, domain).

Key features:
- Bayesian uncertainty quantification
- Context-aware routing (uses query features)
- Natural exploration via posterior sampling
- Sliding window for non-stationarity adaptation

When to use:
- ✅ You want Bayesian uncertainty quantification with contextual features
- ✅ Cold start scenarios (works well with little data)
- ✅ Natural exploration via sampling preferred over UCB
- ❌ You need deterministic decisions (use LinUCB instead)
- ❌ Computational cost is critical (sampling requires Cholesky decomposition)
"""

import asyncio
import logging
import os
from pydantic import BaseModel

from conduit.utils.service_factory import create_service
from conduit.engines.bandits import ContextualThompsonSamplingBandit
from conduit.engines.bandits.base import ModelArm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisResult(BaseModel):
    """Example result type for structured outputs."""

    content: str


async def main() -> None:
    """Run contextual Thompson sampling routing example."""
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=sk-...")
        exit(1)

    print("=" * 60)
    print("Contextual Thompson Sampling Routing Example")
    print("=" * 60)

    # Create service (this will initialize with default algorithm)
    service = await create_service(default_result_type=AnalysisResult)

    # Get model arms from service
    arms = list(service.router.bandit.arm_list)

    # Create Contextual Thompson Sampling bandit
    bandit = ContextualThompsonSamplingBandit(
        arms,
        lambda_reg=1.0,      # Regularization parameter (higher = more stable)
        window_size=1000,     # Sliding window for non-stationarity
        random_seed=42,       # For reproducibility
        success_threshold=0.85,  # Reward threshold for statistics
    )

    # Replace service's bandit with our Contextual Thompson Sampling version
    service.router.bandit = bandit

    print("\nAlgorithm Configuration:")
    print(f"  - Algorithm: Contextual Thompson Sampling (Bayesian)")
    print(f"  - Lambda (regularization): {bandit.lambda_reg}")
    print(f"  - Window size: {bandit.window_size}")
    print(f"  - Feature dimensions: {bandit.feature_dim}")
    print(f"  - Success threshold: {bandit.success_threshold}")

    # Test queries with different characteristics
    test_queries = [
        {
            "query": "What is 2+2? Explain briefly.",
            "description": "Simple math (low complexity)",
        },
        {
            "query": "Explain quantum entanglement and its implications for quantum computing in detail.",
            "description": "Complex technical (high complexity)",
        },
        {
            "query": "Write a Python function to reverse a string.",
            "description": "Code generation (medium complexity)",
        },
    ]

    print(f"\n{'=' * 60}")
    print("Running Test Queries")
    print(f"{'=' * 60}\n")

    for i, test in enumerate(test_queries, 1):
        query_text = test["query"]
        description = test["description"]

        print(f"Query {i}: {description}")
        print(f"Text: {query_text[:60]}...")

        try:
            # Route and execute query
            result = await service.complete(
                prompt=query_text,
                user_id="example_user",
            )

            # Get current statistics
            stats = bandit.get_stats()

            print(f"\nResults:")
            print(f"  - Selected Model: {result.model}")
            print(f"  - Cost: ${result.metadata.get('cost', 0.0):.6f}")
            print(f"  - Latency: {result.metadata.get('latency', 0.0):.2f}s")
            print(f"  - Response length: {len(result.data.get('content', ''))} chars")

            # Show posterior statistics for selected model
            selected_model = result.model
            print(f"\nPosterior Statistics (for {selected_model}):")
            print(f"  - Posterior mean norm: {stats['arm_mu_norms'].get(selected_model, 0.0):.4f}")
            print(f"  - Posterior uncertainty (trace): {stats['arm_sigma_traces'].get(selected_model, 0.0):.2f}")
            print(f"  - Arm pulls: {stats['arm_pulls'].get(selected_model, 0)}")
            print(f"  - Success rate: {stats['arm_success_rates'].get(selected_model, 0.0):.2%}")

        except Exception as e:
            logger.error(f"Error executing query {i}: {e}")
            print(f"\nError: {e}")

        print(f"\n{'-' * 60}\n")

    # Final statistics across all models
    print(f"{'=' * 60}")
    print("Final Algorithm Statistics")
    print(f"{'=' * 60}\n")

    final_stats = bandit.get_stats()
    print(f"Total queries: {final_stats['total_queries']}")
    print(f"\nPer-model statistics:")

    for model_id in arms:
        pulls = final_stats['arm_pulls'].get(model_id, 0)
        success_rate = final_stats['arm_success_rates'].get(model_id, 0.0)
        mu_norm = final_stats['arm_mu_norms'].get(model_id, 0.0)
        sigma_trace = final_stats['arm_sigma_traces'].get(model_id, 0.0)

        print(f"\n{model_id}:")
        print(f"  - Pulls: {pulls}")
        print(f"  - Success rate: {success_rate:.2%}")
        print(f"  - Posterior mean norm: {mu_norm:.4f}")
        print(f"  - Posterior uncertainty: {sigma_trace:.2f}")

    print(f"\n{'=' * 60}")
    print("How Contextual Thompson Sampling Works")
    print(f"{'=' * 60}\n")

    print("1. **Bayesian Linear Regression**:")
    print("   - Maintains posterior distribution: θ ~ N(μ, Σ) for each model")
    print("   - μ (mu): Posterior mean (expected reward coefficients)")
    print("   - Σ (Sigma): Posterior covariance (uncertainty)")
    print()
    print("2. **Selection Process**:")
    print("   - Sample θ_hat from N(μ, Σ) for each model")
    print("   - Compute expected reward: r = θ_hat^T · x")
    print("   - Select model with highest sampled reward")
    print()
    print("3. **Learning**:")
    print("   - Starts with high uncertainty (Σ = I)")
    print("   - Uncertainty decreases with observations")
    print("   - Automatically balances exploration/exploitation")
    print()
    print("4. **Non-stationarity Handling**:")
    print(f"   - Sliding window size: {bandit.window_size} observations")
    print("   - Adapts to model quality/cost changes over time")
    print("   - Only uses recent observations for posterior")
    print()
    print("5. **Advantages**:")
    print("   - Natural Bayesian uncertainty quantification")
    print("   - Works well with little data (cold start)")
    print("   - Uses query context (embedding, complexity, domain)")
    print("   - Probabilistic reward estimates")
    print()

    # Cleanup
    await service.database.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
