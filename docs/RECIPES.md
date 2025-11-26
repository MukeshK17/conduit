# Common Recipes

Copy-paste solutions for common Conduit Router use cases.

---

## Quick Navigation

| I want to... | Jump to |
|--------------|---------|
| Get started in 5 minutes | [Minimal Setup](#minimal-setup) |
| Optimize for cost | [Cost-First Routing](#cost-first-routing) |
| Maximize quality | [Quality-First Routing](#quality-first-routing) |
| Use different strategies per query | [Per-Query Optimization](#per-query-optimization) |
| Integrate with LiteLLM | [LiteLLM Integration](#litellm-integration) |
| Add quality monitoring | [LLM-as-Judge Evaluation](#llm-as-judge-evaluation) |
| Speed up cold start | [Fast Cold Start](#fast-cold-start) |
| A/B test models | [A/B Testing Models](#ab-testing-models) |
| Gradually roll out new model | [Gradual Rollout](#gradual-model-rollout) |
| Debug routing decisions | [Debugging Routing](#debugging-routing-decisions) |

---

## Minimal Setup

**Time**: 2 minutes

```python
# 1. Create .env file
# OPENAI_API_KEY=sk-...

# 2. Run this code
import asyncio
from conduit.engines.router import Router
from conduit.core.models import Query

async def main():
    router = Router()
    decision = await router.route(Query(text="What is 2+2?"))
    print(f"Model: {decision.selected_model}")
    print(f"Confidence: {decision.confidence:.0%}")

asyncio.run(main())
```

**Expected output**:
```
Model: o4-mini
Confidence: 73%
```

---

## Cost-First Routing

**Goal**: Cheapest model that meets minimum quality threshold.

### Option 1: Per-Query Preference

```python
from conduit.core.models import Query, UserPreferences

query = Query(
    text="What's the capital of France?",
    preferences=UserPreferences(optimization_target="cost")
)
decision = await router.route(query)
# Routes to cheapest model (likely o4-mini or gemini-flash)
```

### Option 2: Default for All Queries

```yaml
# conduit.yaml
routing:
  default_optimization: cost
```

```python
router = Router()  # Will use cost optimization by default
```

### Option 3: Custom Weights

```python
# 50% cost weight (vs 20% default)
import os
os.environ["REWARD_WEIGHT_QUALITY"] = "0.4"
os.environ["REWARD_WEIGHT_COST"] = "0.5"
os.environ["REWARD_WEIGHT_LATENCY"] = "0.1"

router = Router()
```

---

## Quality-First Routing

**Goal**: Best possible response, cost is secondary.

```python
from conduit.core.models import Query, UserPreferences

query = Query(
    text="Explain the implications of quantum entanglement for cryptography",
    preferences=UserPreferences(optimization_target="quality")
)
decision = await router.route(query)
# Routes to highest-quality model (likely claude-opus or gpt-5.1)
```

---

## Per-Query Optimization

**Goal**: Different strategies for different query types.

```python
from conduit.core.models import Query, UserPreferences

async def smart_route(text: str, priority: str = "balanced"):
    """Route with appropriate optimization strategy."""
    query = Query(
        text=text,
        preferences=UserPreferences(optimization_target=priority)
    )
    return await router.route(query)

# Simple questions → optimize for cost
decision = await smart_route("What is 2+2?", priority="cost")

# Complex analysis → optimize for quality
decision = await smart_route(
    "Compare the economic policies of Keynes and Hayek",
    priority="quality"
)

# User waiting → optimize for speed
decision = await smart_route("Quick summary of this text", priority="speed")
```

---

## LiteLLM Integration

**Goal**: Use Conduit's ML routing with LiteLLM's 100+ providers.

```python
from litellm import Router as LiteLLMRouter
from conduit_litellm import ConduitRoutingStrategy

# Configure LiteLLM models
litellm_router = LiteLLMRouter(
    model_list=[
        {"model_name": "gpt-4", "litellm_params": {"model": "gpt-4o-mini"}},
        {"model_name": "claude", "litellm_params": {"model": "claude-3-haiku-20240307"}},
        {"model_name": "gemini", "litellm_params": {"model": "gemini/gemini-1.5-flash"}},
    ]
)

# Add Conduit's ML routing
strategy = ConduitRoutingStrategy(use_hybrid=True)
ConduitRoutingStrategy.setup_strategy(litellm_router, strategy)

# LiteLLM now uses Conduit's intelligent routing
response = await litellm_router.acompletion(
    model="gpt-4",  # Conduit picks actual model
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## LLM-as-Judge Evaluation

**Goal**: Automatically evaluate response quality using another LLM.

```python
from conduit_litellm import ConduitRoutingStrategy
from conduit.evaluation import ArbiterEvaluator

# Create evaluator (samples 10% of responses)
evaluator = ArbiterEvaluator(
    sample_rate=0.1,      # Evaluate 10% of responses
    daily_budget=10.0,    # Max $10/day on evaluations
    model="o4-mini"       # Cheap model for judging
)

# Attach to routing strategy
strategy = ConduitRoutingStrategy(
    use_hybrid=True,
    evaluator=evaluator
)

# Evaluator automatically runs in background
# Quality scores fed back to bandit for learning
```

---

## Fast Cold Start

**Goal**: Production-quality routing in 2,000-3,000 queries instead of 10,000+.

### Enable Hybrid Routing + PCA

```python
from conduit.engines.hybrid_router import HybridRouter

router = HybridRouter(
    use_pca=True,           # 75% sample reduction
    pca_components=67,      # Reduce 387 dims to 67
    switch_threshold=2000   # Switch to LinUCB after 2000 queries
)

# 30% faster convergence than pure LinUCB
```

### With Informed Priors

```yaml
# conduit.yaml - give the bandit a head start
priors:
  code:
    claude-sonnet-4.5: 0.92  # We know Claude is good at code
    gpt-5.1: 0.88
  general:
    gpt-5.1: 0.88
    o4-mini: 0.78
```

---

## A/B Testing Models

**Goal**: Compare two models head-to-head.

```python
import random
from conduit.engines.router import Router
from conduit.core.models import Query

class ABTestRouter:
    def __init__(self, model_a: str, model_b: str, split: float = 0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split = split
        self.results = {"a": [], "b": []}

    async def route(self, query: Query):
        # Random assignment
        if random.random() < self.split:
            model = self.model_a
            group = "a"
        else:
            model = self.model_b
            group = "b"

        return model, group

    def record_result(self, group: str, quality: float, cost: float):
        self.results[group].append({"quality": quality, "cost": cost})

    def get_stats(self):
        for group in ["a", "b"]:
            results = self.results[group]
            if results:
                avg_quality = sum(r["quality"] for r in results) / len(results)
                avg_cost = sum(r["cost"] for r in results) / len(results)
                print(f"Model {group}: {len(results)} queries, "
                      f"quality={avg_quality:.2f}, cost=${avg_cost:.4f}")

# Usage
ab_router = ABTestRouter("gpt-5.1", "claude-sonnet-4.5")
model, group = await ab_router.route(query)
# ... execute query ...
ab_router.record_result(group, quality=0.9, cost=0.001)
ab_router.get_stats()
```

---

## Gradual Model Rollout

**Goal**: Test a new model with increasing traffic before full deployment.

```python
import random
from datetime import datetime, timedelta

class GradualRollout:
    def __init__(
        self,
        new_model: str,
        fallback_model: str,
        start_percent: float = 5.0,
        target_percent: float = 100.0,
        ramp_days: int = 7
    ):
        self.new_model = new_model
        self.fallback_model = fallback_model
        self.start_percent = start_percent
        self.target_percent = target_percent
        self.ramp_days = ramp_days
        self.start_time = datetime.now()

    def get_current_percent(self) -> float:
        """Calculate current rollout percentage."""
        elapsed = datetime.now() - self.start_time
        progress = min(1.0, elapsed.total_seconds() / (self.ramp_days * 86400))
        return self.start_percent + (self.target_percent - self.start_percent) * progress

    def select_model(self) -> str:
        """Select model based on current rollout percentage."""
        if random.random() * 100 < self.get_current_percent():
            return self.new_model
        return self.fallback_model

# Usage: Roll out gpt-5.1 over 7 days
rollout = GradualRollout(
    new_model="gpt-5.1",
    fallback_model="gpt-4o",
    start_percent=5.0,    # Start at 5%
    target_percent=100.0, # End at 100%
    ramp_days=7           # Over 7 days
)

model = rollout.select_model()
print(f"Current rollout: {rollout.get_current_percent():.1f}%")
print(f"Selected: {model}")
```

---

## Debugging Routing Decisions

**Goal**: Understand why Conduit made a specific routing decision.

### Check Bandit Statistics

```python
router = Router()

# Get detailed statistics
stats = router.bandit.get_stats()

print(f"Algorithm: {stats['name']}")
print(f"Total queries: {stats['total_queries']}")
print()

# Per-model stats
if 'arm_pulls' in stats:
    print("Model Statistics:")
    for model_id, pulls in stats['arm_pulls'].items():
        mean_reward = stats.get('arm_mean_rewards', {}).get(model_id, 0.0)
        print(f"  {model_id}:")
        print(f"    Pulls: {pulls}")
        print(f"    Mean reward: {mean_reward:.3f}")
```

### Inspect a Single Decision

```python
from conduit.core.models import Query

query = Query(text="Write a Python function to sort a list")
decision = await router.route(query)

print(f"Selected: {decision.selected_model}")
print(f"Confidence: {decision.confidence:.2%}")
print(f"Reasoning: {decision.reasoning}")
print(f"Features:")
print(f"  Complexity: {decision.features.complexity_score:.2f}")
print(f"  Domain: {decision.features.domain}")
print(f"  Tokens: {decision.features.token_count}")
```

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('conduit.engines.router').setLevel(logging.DEBUG)
logging.getLogger('conduit.engines.bandits').setLevel(logging.DEBUG)

# Now routing will log detailed decisions
router = Router()
decision = await router.route(query)
```

---

## Comparing Algorithms

**Goal**: See how different algorithms route the same query.

```python
from conduit.engines.router import Router
from conduit.engines.bandits import (
    LinUCBBandit,
    UCB1Bandit,
    ThompsonSamplingBandit,
    EpsilonGreedyBandit
)
from conduit.core.models import Query

query = Query(text="Explain machine learning")

algorithms = {
    "LinUCB": "linucb",
    "UCB1": "ucb1",
    "Thompson": "thompson_sampling",
    "Epsilon-Greedy": "epsilon_greedy"
}

print("Routing comparison for: 'Explain machine learning'\n")

for name, alg in algorithms.items():
    router = Router(algorithm=alg)
    decision = await router.route(query)
    print(f"{name:15} → {decision.selected_model:20} (confidence: {decision.confidence:.0%})")
```

**Expected output**:
```
Routing comparison for: 'Explain machine learning'

LinUCB          → gpt-5.1              (confidence: 82%)
UCB1            → o4-mini              (confidence: 67%)
Thompson        → claude-sonnet-4.5    (confidence: 78%)
Epsilon-Greedy  → gpt-5.1              (confidence: 71%)
```

---

## Production Checklist

Before deploying to production:

```python
# 1. Verify configuration loads
from conduit.core.config import load_config
config = load_config()
print(f"Default optimization: {config.routing.default_optimization}")

# 2. Check available models
from conduit.models import available_models
models = available_models()
print(f"Available models: {len(models)}")
for m in models[:5]:
    print(f"  - {m.model_id}")

# 3. Test routing works
from conduit.engines.router import Router
from conduit.core.models import Query

router = Router()
decision = await router.route(Query(text="Test query"))
print(f"Routing works: {decision.selected_model}")

# 4. Check Redis (if using)
# If Redis is down, Conduit gracefully degrades
print(f"Cache status: {'connected' if router.cache else 'disabled'}")

# 5. Run a few test queries
test_queries = [
    "Simple question",
    "Complex technical analysis needed here",
    "Write code for quicksort"
]
for q in test_queries:
    d = await router.route(Query(text=q))
    print(f"'{q[:30]}...' → {d.selected_model}")
```

---

## See Also

- [CONFIGURATION.md](CONFIGURATION.md) - All configuration options
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Debug common issues
- [FAQ.md](FAQ.md) - Frequently asked questions
- [BANDIT_ALGORITHMS.md](BANDIT_ALGORITHMS.md) - Algorithm deep dive
