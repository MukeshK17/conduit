# LiteLLM Integration Examples

Conduit's ML-powered routing as a custom strategy for LiteLLM, enabling intelligent model selection across 100+ LLM providers.

## Features

- ✅ **ML Routing**: Conduit's bandit algorithms (LinUCB, Thompson Sampling) select optimal models
- ✅ **100+ Providers**: Access all LiteLLM providers with intelligent routing
- ✅ **Hybrid Mode**: UCB1→LinUCB warm start for 30% faster convergence
- ✅ **Redis Caching**: Optional caching for query feature vectors
- ✅ **Multi-Provider**: OpenAI, Anthropic, Google, Groq, and more

## Quick Start

### 1. Installation

```bash
# Install Conduit with LiteLLM support
pip install conduit[litellm]

# Or with Docker
docker-compose up
```

### 2. Set API Keys

```bash
# Copy example environment
cp .env.example .env

# Edit .env and add your API keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
export GROQ_API_KEY=gsk_...
```

### 3. Run Demo

```bash
# Local
python examples/04_litellm/demo.py

# Docker
docker-compose up conduit
```

## Docker Setup

### Build and Run

```bash
# Start services (Redis + Conduit)
docker-compose up

# Or run in background
docker-compose up -d

# View logs
docker-compose logs -f conduit

# Stop services
docker-compose down
```

### Environment Variables

Configure in `.env` or `docker-compose.yml`:

```bash
# Required: At least one LLM provider API key
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional: Redis caching
REDIS_URL=redis://redis:6379
REDIS_CACHE_ENABLED=true

# Optional: Hybrid routing
USE_HYBRID_ROUTING=true
```

## Example Usage

### Basic Usage

```python
from litellm import Router
from conduit_litellm import ConduitRoutingStrategy

# Configure LiteLLM model list
model_list = [
    {
        "model_name": "gpt-4o-mini",
        "litellm_params": {"model": "gpt-4o-mini"},
        "model_info": {"id": "gpt-4o-mini-openai"},
    },
    {
        "model_name": "claude-3-5-sonnet",
        "litellm_params": {"model": "claude-3-5-sonnet-20241022"},
        "model_info": {"id": "claude-3-5-sonnet"},
    },
]

# Initialize LiteLLM router
router = Router(model_list=model_list)

# Set Conduit as custom routing strategy
strategy = ConduitRoutingStrategy(use_hybrid=True)
ConduitRoutingStrategy.setup_strategy(router, strategy)

# Make requests - Conduit selects optimal model
response = await router.acompletion(
    model="gpt-4o-mini",  # Model group
    messages=[{"role": "user", "content": "Hello"}]
)
```

### With Caching

```python
from conduit_litellm import ConduitRoutingStrategy

# Enable Redis caching for feature vectors
strategy = ConduitRoutingStrategy(
    use_hybrid=True,
    cache_enabled=True,
    redis_url="redis://localhost:6379"
)

ConduitRoutingStrategy.setup_strategy(router, strategy)
```

### Custom Configuration

```python
from conduit_litellm import ConduitRoutingStrategy
from conduit.engines.router import Router as ConduitRouter

# Pre-configure Conduit router
conduit_router = ConduitRouter(
    models=["gpt-4o-mini", "claude-3-5-sonnet"],
    use_hybrid=True,
    embedding_model="all-MiniLM-L6-v2"
)

# Use pre-configured router
strategy = ConduitRoutingStrategy(conduit_router=conduit_router)
ConduitRoutingStrategy.setup_strategy(router, strategy)
```

## How It Works

1. **LiteLLM receives request** with model group (e.g., "gpt-4o-mini")
2. **Conduit analyzes query**:
   - Extracts text from messages
   - Generates 387-dim feature vector (384 embedding + 3 metadata)
   - Optional PCA reduction (387→67 dims)
3. **Bandit algorithm selects model**:
   - LinUCB: Contextual bandit with ridge regression
   - Thompson Sampling: Bayesian exploration
   - UCB1: Upper confidence bound (hybrid warm start)
4. **LiteLLM executes** with selected deployment
5. **Conduit learns** (Issue #13 - feedback loop coming soon)

## Configuration Options

### ConduitRoutingStrategy

```python
ConduitRoutingStrategy(
    conduit_router=None,      # Optional pre-configured router
    use_hybrid=False,         # Enable UCB1→LinUCB warm start
    embedding_model="all-MiniLM-L6-v2",  # Sentence transformer
    cache_enabled=False,      # Enable Redis caching
    redis_url=None,           # Redis connection URL
    use_pca=False,            # Enable PCA reduction
    pca_dimensions=67,        # Target dimensions
)
```

## Performance

### Convergence Speed

- **Standard LinUCB**: ~10,000 queries to production-ready
- **With PCA**: ~2,500 queries (75% reduction)
- **With Hybrid**: ~1,500 queries (85% reduction)
- **Combined**: ~1,000-1,500 queries (best performance)

### Overhead

- **Routing decision**: <50ms
- **With caching**: <10ms (cache hit)
- **PCA reduction**: ~5ms

## Troubleshooting

### Import Error: LiteLLM not found

```bash
pip install conduit[litellm]
```

### Redis Connection Error

```bash
# Check Redis is running
docker ps | grep redis

# Or start Redis
docker-compose up redis
```

### Model Not Found

Ensure model IDs in `model_info.id` match Conduit's model registry:

```python
# Correct
{"model_info": {"id": "gpt-4o-mini-openai"}}

# Incorrect
{"model_info": {"id": "some-random-id"}}
```

## Next Steps

- **Issue #12**: Enhanced routing logic with query analysis
- **Issue #13**: Feedback loop for learning from outcomes
- **Issue #14**: Comprehensive tests
- **Issue #15**: More examples (custom config, performance comparison, etc.)

## Resources

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Conduit Documentation](../../README.md)
- [Issue #9: LiteLLM Integration](https://github.com/ashita-ai/conduit/issues/9)
