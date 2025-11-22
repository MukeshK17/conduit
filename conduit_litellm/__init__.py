"""Conduit LiteLLM Plugin - ML-powered routing strategy for LiteLLM.

This plugin integrates Conduit's ML-based routing algorithms with LiteLLM's
router, enabling intelligent model selection across 100+ LLM providers.

Usage:
    from litellm import Router
    from conduit_litellm import ConduitRoutingStrategy

    router = Router(model_list=[...])
    router.set_custom_routing_strategy(ConduitRoutingStrategy())
"""

from conduit_litellm.strategy import ConduitRoutingStrategy

__version__ = "0.1.0"
__all__ = ["ConduitRoutingStrategy"]
