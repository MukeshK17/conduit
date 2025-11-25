# LiteLLM Integration Examples

ML-powered intelligent routing for LiteLLM using Conduit's contextual bandit algorithms.

## Overview

These examples demonstrate how to use Conduit as a custom routing strategy for LiteLLM, enabling intelligent model selection across 100+ LLM providers based on query features, cost, and quality.

## ⚠️ Key Concept: Shared Model Names

**IMPORTANT**: For Conduit routing to work, multiple model deployments must use the **SAME `model_name`**.

### ❌ Wrong (No Routing)

```python
model_list = [
    {"model_name": "gpt-4o-mini", "litellm_params": {"model": "gpt-4o-mini"}, ...},  # Unique name
    {"model_name": "gpt-4o", "litellm_params": {"model": "gpt-4o"}, ...},           # Different name
]

# This bypasses Conduit - goes directly to gpt-4o-mini
response = await router.acompletion(model="gpt-4o-mini", messages=[...])
```

### ✅ Right (Conduit Routes)

```python
model_list = [
    {"model_name": "gpt", "litellm_params": {"model": "gpt-4o-mini"}, ...},  # SAME name
    {"model_name": "gpt", "litellm_params": {"model": "gpt-4o"}, ...},       # SAME name
]

# Conduit intelligently selects between gpt-4o-mini and gpt-4o
response = await router.acompletion(model="gpt", messages=[...])
```

**Why this matters**: When you use a shared `model_name`, LiteLLM delegates model selection to Conduit's ML routing strategy. Otherwise, LiteLLM directly routes to the exact model you specified.

## Installation

```bash
pip install conduit[litellm]
```

## API Keys Required

Set at least one provider API key:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export GROQ_API_KEY="your-key"
```

## Examples

### 1. Basic Usage (`basic_usage.py`)

**Simplest example** showing Conduit + LiteLLM integration with a single provider.

```bash
python examples/04_litellm/basic_usage.py
```

**What it shows:**
- Shared `model_name` pattern for routing (both models use "gpt")
- Routes between gpt-4o-mini (cheap) and gpt-4o (capable)
- Learns which model is best for simple vs complex queries
- 5 diverse test queries to demonstrate learning

**Use this when:** You're new to Conduit and want the simplest working example.

---

### 2. Multi-Provider Routing (`multi_provider.py`)

**Cross-provider routing** across OpenAI, Anthropic, Google, and Groq.

```bash
python examples/04_litellm/multi_provider.py
```

**What it shows:**
- All models share "llm" as `model_name` (enables cross-provider routing)
- Conduit picks the best provider+model for each query type
- Cost optimization across multiple providers
- Automatic provider fallback if one is unavailable

**Use this when:** You have multiple provider API keys and want Conduit to route across all of them.

---

### 3. Complete Demo (`demo.py`)

**Full-featured example** with detailed logging and error handling.

```bash
python examples/04_litellm/demo.py
```

**What it shows:**
- Provider detection and configuration
- Detailed status messages at each step
- Error handling and graceful degradation
- Complete integration workflow

**Use this when:** You want to understand the full integration flow with comprehensive logging.

---

### 4. Custom Configuration (`custom_config.py`)

**Advanced configuration** showing customization options.

```bash
python examples/04_litellm/custom_config.py
```

**What it shows:**
- Hybrid routing (UCB1 → LinUCB warm start, always enabled by default)
- Redis caching integration (optional)
- Custom embedding models (optional)
- Cost tracking and optimization

**Use this when:** You need to tune performance, enable caching, or customize behavior.

---

### 5. Arbiter Quality Measurement (`arbiter_quality_measurement.py`)

**LLM-as-judge quality evaluation** using Arbiter evaluator.

```bash
python examples/04_litellm/arbiter_quality_measurement.py
```

**What it shows:**
- Shared "llm" model_name for routing across OpenAI and Anthropic
- Fire-and-forget async quality evaluation (doesn't block routing)
- Configurable sampling rate (10% default) to control costs
- Automatic feedback storage for bandit learning
- Graceful degradation if evaluation fails

**Use this when:** You want automated quality measurement using LLM-as-judge instead of just implicit signals.

**Requirements:**
- DATABASE_URL environment variable (PostgreSQL)
- OPENAI_API_KEY or ANTHROPIC_API_KEY

## How It Works

### 1. Configure Model List with Shared Names

```python
from litellm import Router
from conduit_litellm import ConduitRoutingStrategy

# KEY: Multiple models with SAME model_name for Conduit to route between them
model_list = [
    {
        "model_name": "gpt",  # Shared name - Conduit picks between these
        "litellm_params": {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        "model_info": {"id": "gpt-4o-mini"},  # Required for Conduit
    },
    {
        "model_name": "gpt",  # SAME name - part of routing pool
        "litellm_params": {
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        "model_info": {"id": "gpt-4o"},  # Required for Conduit
    },
]

# Initialize LiteLLM router
router = Router(model_list=model_list)
```

### 2. Setup Conduit Strategy

```python
# Create Conduit strategy (hybrid routing always enabled)
strategy = ConduitRoutingStrategy()

# Activate Conduit routing
ConduitRoutingStrategy.setup_strategy(router, strategy)
```

### 3. Make Requests

```python
# Use the shared model_name - Conduit selects optimal model
response = await router.acompletion(
    model="gpt",  # Conduit intelligently chooses between gpt-4o-mini and gpt-4o
    messages=[{"role": "user", "content": "Your query"}]
)
```

### 4. Automatic Learning

Conduit automatically:
- Extracts query features (embeddings, complexity, domain)
- Selects optimal model using bandit algorithm (UCB1 → LinUCB)
- Learns from response (cost, latency, quality)
- Improves future routing decisions

No manual rules, no configuration files, just ML-powered intelligence.

## Common Patterns

### Single Provider, Multiple Models

```python
# Both models from OpenAI, shared "gpt" name
model_list = [
    {"model_name": "gpt", "litellm_params": {"model": "gpt-4o-mini"}, ...},
    {"model_name": "gpt", "litellm_params": {"model": "gpt-4o"}, ...},
]
response = await router.acompletion(model="gpt", ...)  # Conduit routes
```

### Multi-Provider, Shared Pool

```python
# All providers in one pool, shared "llm" name
model_list = [
    {"model_name": "llm", "litellm_params": {"model": "gpt-4o-mini"}, ...},      # OpenAI
    {"model_name": "llm", "litellm_params": {"model": "claude-3-5-haiku"}, ...}, # Anthropic
    {"model_name": "llm", "litellm_params": {"model": "gemini-2.5-flash"}, ...}, # Google
]
response = await router.acompletion(model="llm", ...)  # Routes across all providers
```

### Separate Pools for Different Use Cases

```python
# Separate pools for different query types
model_list = [
    # Fast pool
    {"model_name": "fast", "litellm_params": {"model": "gpt-4o-mini"}, ...},
    {"model_name": "fast", "litellm_params": {"model": "claude-3-5-haiku"}, ...},

    # Quality pool
    {"model_name": "quality", "litellm_params": {"model": "gpt-4o"}, ...},
    {"model_name": "quality", "litellm_params": {"model": "claude-3-5-sonnet"}, ...},
]

# Route to appropriate pool
fast_response = await router.acompletion(model="fast", ...)
quality_response = await router.acompletion(model="quality", ...)
```

## Configuration Options

### Hybrid Routing (Always Enabled)

Hybrid routing (UCB1→LinUCB warm start) is always enabled by default. It achieves 30% faster convergence by:
- Starting with UCB1 (fast exploration)
- Switching to LinUCB after ~100 queries (contextual optimization)

### Redis Caching

```python
strategy = ConduitRoutingStrategy(
    cache_enabled=True,
    redis_url="redis://localhost:6379"
)
```

Caches query embeddings and routing decisions for performance.

### Custom Embedding Model

```python
strategy = ConduitRoutingStrategy(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
```

Change the sentence transformer model for different trade-offs (speed vs accuracy).

## Feedback Loop

Conduit automatically learns from LiteLLM responses through `ConduitFeedbackLogger`:

- **Cost**: Extracted from `response._hidden_params['response_cost']`
- **Latency**: Measured from request timing
- **Quality**: Estimated from success/failure (0.9 for success, 0.1 for errors)

The feedback loop updates the bandit algorithm, improving future routing decisions.

## Performance

### Hybrid Routing Convergence

- **First 100 queries**: UCB1 explores all models quickly
- **After 100 queries**: LinUCB uses query context for optimal selection
- **Result**: 30% faster convergence vs pure LinUCB

### Cost Savings

Typical savings with Conduit vs random selection:
- **30-50% cost reduction** by routing simple queries to cheaper models
- **Quality maintained** by routing complex queries to powerful models

## Bootstrap Process

Conduit uses a 3-tier fallback system for cold start:

1. **YAML Config** (`conduit.yaml`): Default costs and quality estimates
2. **Environment Variables**: Override YAML for specific deployments
3. **Hardcoded Defaults**: Final fallback if nothing else configured

### Hybrid Routing Timeline

- **Queries 1-100**: UCB1 (fast exploration, ignores query context)
- **Queries 100+**: LinUCB (contextual optimization, uses embeddings)
- **Result**: 30% faster convergence vs pure LinUCB

### Pricing Updates

Conduit learns actual costs from LiteLLM's `response._hidden_params['response_cost']`. The bootstrap pricing in YAML is only used for initial routing decisions before real data is collected.

**Best practice**: Update `conduit.yaml` pricing monthly from learned data for better cold starts on new deployments.

## Troubleshooting

### "Conduit always picks the same model"

**Cause**: Each model has a unique `model_name`, so LiteLLM bypasses Conduit.

**Fix**: Use shared `model_name` for all models you want Conduit to route between:
```python
# ❌ Wrong - each model has unique name
model_list = [
    {"model_name": "gpt-4o-mini", ...},  # Unique
    {"model_name": "gpt-4o", ...},       # Different
]

# ✅ Right - shared name enables routing
model_list = [
    {"model_name": "gpt", ...},  # Same
    {"model_name": "gpt", ...},  # Same
]
```

### "LiteLLM not installed"

```bash
pip install conduit[litellm]
```

### "No API keys found"

Set at least one:
```bash
export OPENAI_API_KEY="your-key"
```

### "Redis connection failed"

Conduit works without Redis (in-memory mode). To enable caching:
```bash
docker run -d -p 6379:6379 redis
```

### "Model not found"

Ensure your LiteLLM `model_list` includes `model_info.id` for each model:
```python
{
    "model_name": "gpt",  # Shared name for routing
    "litellm_params": {...},
    "model_info": {"id": "gpt-4o-mini"}  # Required for Conduit
}
```

### "Routing seems random in early queries"

**Expected behavior**: The first ~100 queries use UCB1 (non-contextual) for fast exploration. This may seem random but it's systematically exploring all models to gather performance data.

**After 100 queries**: Conduit switches to LinUCB which uses query context (embeddings, complexity) for smarter routing.

## Next Steps

- **Production deployment**: See `docs/LITELLM_INTEGRATION.md`
- **Custom algorithms**: See `docs/BANDIT_ALGORITHMS.md`
- **Advanced features**: See `examples/03_optimization/`

## Related Issues

- #13: LiteLLM feedback loop (✅ Implemented)
- #14: LiteLLM examples (this directory)
- #15: LiteLLM documentation
- #16: LiteLLM plugin announcement
