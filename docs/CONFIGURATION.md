# Configuration Reference

**Single source of truth** for all Conduit Router configuration options.

---

## Quick Reference

| I want to... | Configure via |
|--------------|---------------|
| Set API keys | `.env` file |
| Change reward weights | `conduit.yaml` → `routing.presets` |
| Adjust algorithm behavior | `conduit.yaml` → `algorithms` |
| Set model priors | `conduit.yaml` → `priors` |
| Override at runtime | Constructor parameters or `UserPreferences` |

---

## Environment Variables

Create a `.env` file in your project root:

### Required (at least one LLM provider)

```bash
# LLM Providers - set at least one
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
GROQ_API_KEY=...
MISTRAL_API_KEY=...
COHERE_API_KEY=...
```

### Embedding Configuration

```bash
# Embedding provider (default: huggingface - free, no key needed)
EMBEDDING_PROVIDER=huggingface    # Options: huggingface, openai, cohere, sentence-transformers

# Optional: Custom embedding model
EMBEDDING_MODEL=                  # Leave empty for provider default

# Provider-specific keys (only if using that provider)
HF_TOKEN=hf_...                   # HuggingFace (optional, increases rate limits)
OPENAI_API_KEY=sk-...             # OpenAI (reuses LLM key)
COHERE_API_KEY=...                # Cohere (separate key required)
```

### Infrastructure (Optional)

```bash
# Database - enables routing history persistence
DATABASE_URL=postgresql://user:pass@localhost:5432/conduit

# Redis - enables caching (10-40x faster repeated queries)
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
```

### Runtime Overrides

```bash
# Override default optimization strategy
ROUTING_DEFAULT_OPTIMIZATION=balanced  # balanced, quality, cost, speed

# Override reward weights (must sum to 1.0)
REWARD_WEIGHT_QUALITY=0.70
REWARD_WEIGHT_COST=0.20
REWARD_WEIGHT_LATENCY=0.10
```

---

## conduit.yaml

Main configuration file. All sections are optional - sensible defaults are used.

### Routing Presets

```yaml
routing:
  # Default optimization when no preference specified
  default_optimization: balanced  # balanced, quality, cost, speed

  # Reward weight presets (weights must sum to 1.0)
  presets:
    balanced:
      quality: 0.7
      cost: 0.2
      latency: 0.1

    quality:
      quality: 0.8
      cost: 0.1
      latency: 0.1

    cost:
      quality: 0.4
      cost: 0.5
      latency: 0.1

    speed:
      quality: 0.4
      cost: 0.1
      latency: 0.5
```

### Algorithm Hyperparameters

```yaml
algorithms:
  linucb:
    alpha: 1.0              # Exploration parameter (higher = more exploration)
    success_threshold: 0.85 # Reward threshold for "success"

  epsilon_greedy:
    epsilon: 0.1            # Exploration rate (10% random)
    decay: 1.0              # Decay rate (1.0 = no decay)
    min_epsilon: 0.01       # Minimum exploration floor

  ucb1:
    c: 1.5                  # Confidence multiplier

  thompson_sampling:
    lambda: 1.0             # Regularization (prior strength)
```

### Context-Specific Priors

Cold start optimization - helps routing before enough data is collected:

```yaml
priors:
  code:
    claude-sonnet-4.5: 0.92  # 92% expected quality for code tasks
    gpt-5.1: 0.88
    o4-mini: 0.78

  creative:
    claude-opus-4.5: 0.94
    claude-sonnet-4.5: 0.90

  analysis:
    claude-opus-4.5: 0.92
    gpt-5.1: 0.89

  simple_qa:
    o4-mini: 0.90            # Cheap models excel at simple questions
    gemini-2.0-flash: 0.88

  general:
    gpt-5.1: 0.88
    claude-opus-4.5: 0.87
```

### Hybrid Routing

```yaml
hybrid_routing:
  switch_threshold: 2000    # Switch from UCB1 to LinUCB after N queries
  ucb1_c: 1.5               # UCB1 exploration parameter
  linucb_alpha: 1.0         # LinUCB exploration parameter
```

### Quality Estimation

Heuristic quality scoring when no explicit rating provided:

```yaml
quality_estimation:
  base_quality: 0.9         # Default quality for successful responses
  failure_quality: 0.1      # Quality for errors
  min_response_chars: 10    # Minimum valid response length

  penalties:
    short_response: 0.15    # Penalty for very short responses
    repetition: 0.30        # Penalty for repetitive text
    no_keyword_overlap: 0.20  # No keywords matching query
```

### Implicit Feedback Detection

```yaml
feedback:
  retry_detection:
    similarity_threshold: 0.85  # 85% similar = likely retry
    time_window_seconds: 300    # 5-minute window

  weights:
    explicit: 0.7           # Weight on user ratings
    implicit: 0.3           # Weight on behavioral signals
```

### Cache Configuration

```yaml
cache:
  enabled: true
  ttl: 86400                # 24 hours
  timeout: 5                # Redis timeout (seconds)

  circuit_breaker:
    threshold: 5            # Failures before opening
    timeout: 300            # Circuit open duration (5 min)
```

### Arbiter LLM-as-Judge

```yaml
arbiter:
  sample_rate: 0.1          # Evaluate 10% of responses
  daily_budget: 10.0        # Max $10/day on evaluations
  model: "o4-mini"          # Cheap model for judging
  evaluators:
    - semantic
    - factuality
```

---

## Constructor Parameters

Override configuration programmatically:

### Router

```python
from conduit.engines.router import Router

router = Router(
    # Algorithm selection
    algorithm="linucb",           # linucb, ucb1, thompson_sampling, epsilon_greedy

    # Hybrid routing
    use_hybrid_routing=True,      # Enable UCB1→LinUCB warm start

    # Embedding provider
    embedding_provider_type="huggingface",  # huggingface, openai, cohere
    embedding_model=None,         # Provider default if None

    # Model selection
    models=None,                  # Auto-detect from API keys if None
)
```

### HybridRouter

```python
from conduit.engines.hybrid_router import HybridRouter

router = HybridRouter(
    switch_threshold=2000,        # Queries before switching to LinUCB
    use_pca=True,                 # Enable PCA dimensionality reduction
    pca_components=67,            # PCA target dimensions
)
```

### Per-Query Preferences

```python
from conduit.core.models import Query, UserPreferences

# Override optimization per query
query = Query(
    text="Write a poem about the ocean",
    preferences=UserPreferences(
        optimization_target="quality"  # balanced, quality, cost, speed
    )
)

decision = await router.route(query)
```

---

## Configuration Precedence

When the same setting is configured multiple places:

1. **Constructor parameters** (highest priority)
2. **Environment variables**
3. **conduit.yaml**
4. **Built-in defaults** (lowest priority)

Example: Setting reward weights

```python
# These all work, in order of precedence:

# 1. Constructor (wins)
router = Router(reward_weights={"quality": 0.5, "cost": 0.4, "latency": 0.1})

# 2. Environment variable
# REWARD_WEIGHT_QUALITY=0.5

# 3. conduit.yaml
# routing:
#   presets:
#     custom: {quality: 0.5, cost: 0.4, latency: 0.1}

# 4. Built-in default (balanced: 0.7/0.2/0.1)
```

---

## Common Configurations

### Cost-Optimized Deployment

```yaml
# conduit.yaml
routing:
  default_optimization: cost

algorithms:
  linucb:
    alpha: 0.5              # Less exploration (trust known cheap models)
```

### Quality-First Deployment

```yaml
# conduit.yaml
routing:
  default_optimization: quality

priors:
  general:
    claude-opus-4.5: 0.95   # Bias toward premium models
    gpt-5.1: 0.93
```

### Fast Iteration (Development)

```yaml
# conduit.yaml
hybrid_routing:
  switch_threshold: 100     # Faster transition to contextual routing

algorithms:
  epsilon_greedy:
    epsilon: 0.3            # More exploration during development
```

### Production with LLM-as-Judge

```yaml
# conduit.yaml
arbiter:
  sample_rate: 0.1          # Evaluate 10% of responses
  daily_budget: 50.0        # Higher budget for production
  evaluators:
    - semantic
    - factuality
```

---

## Validating Configuration

```python
from conduit.core.config import load_config

# Load and validate configuration
config = load_config()

# Check loaded values
print(f"Default optimization: {config.routing.default_optimization}")
print(f"LinUCB alpha: {config.algorithms.linucb.alpha}")
print(f"Cache enabled: {config.cache.enabled}")
```

---

## See Also

- [EMBEDDING_PROVIDERS.md](EMBEDDING_PROVIDERS.md) - Detailed embedding configuration
- [HYBRID_ROUTING.md](HYBRID_ROUTING.md) - Hybrid routing deep dive
- [PRIORS.md](PRIORS.md) - Context-specific priors explanation
- [BOOTSTRAP_STRATEGY.md](BOOTSTRAP_STRATEGY.md) - Startup and fallback behavior
