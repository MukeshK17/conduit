# Conduit LiteLLM Integration

ML-powered routing strategy for LiteLLM using Conduit's contextual bandits.

## Installation

```bash
pip install conduit[litellm]
```

## Usage

```python
from litellm import Router
from conduit_litellm import ConduitRoutingStrategy

# Configure LiteLLM router with your models
router = Router(
    model_list=[
        {
            "model_name": "gpt-4",
            "litellm_params": {"model": "gpt-4o-mini"},
        },
        {
            "model_name": "claude-3",
            "litellm_params": {"model": "claude-3-haiku"},
        }
    ]
)

# Setup Conduit routing strategy
strategy = ConduitRoutingStrategy(use_hybrid=True)
ConduitRoutingStrategy.setup_strategy(router, strategy)

# Now LiteLLM uses Conduit's ML routing
response = await router.acompletion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Features

- **ML-Powered Selection**: Uses contextual bandits (LinUCB, Thompson Sampling) to learn optimal model routing
- **100+ Providers**: Works with all LiteLLM-supported providers
- **Hybrid Routing**: Optional UCB1→LinUCB warm start for faster convergence
- **Async & Sync**: Supports both async and sync contexts (Issue #31 fixed)

## Configuration Options

```python
strategy = ConduitRoutingStrategy(
    use_hybrid=True,              # Enable UCB1→LinUCB warm start
    cache_enabled=True,           # Enable Redis caching
    redis_url="redis://localhost:6379"
)
```

## Testing

**Note**: LiteLLM is an optional dependency. Tests require `pip install conduit[litellm]`.

```bash
# Run LiteLLM integration tests
pytest tests/unit/test_litellm_strategy.py -v
```

## Issue #31 Fix

The sync `get_available_deployment()` method now correctly handles async contexts by running the async version in a separate thread when an event loop is already running. This prevents the `RuntimeError: This event loop is already running` error.

**Before** (Issue #31):
```python
# Raised RuntimeError in async contexts
return loop.run_until_complete(self.async_get_available_deployment(...))
```

**After** (Fixed):
```python
# Runs in separate thread when event loop exists
with concurrent.futures.ThreadPoolExecutor() as executor:
    future = executor.submit(asyncio.run, self.async_get_available_deployment(...))
    return future.result()
```

## Related Issues

- #9 - LiteLLM integration (parent issue)
- #13 - Feedback collection and learning
- #14 - LiteLLM integration examples
- #15 - LiteLLM plugin usage documentation
- #31 - **Fixed**: RuntimeError in async contexts
