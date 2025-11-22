"""LiteLLM routing strategy using Conduit's ML-powered model selection."""

from typing import Any, Dict, List, Optional, Union

from conduit.core.models import Query
from conduit.engines.router import Router


class ConduitRoutingStrategy:
    """ML-powered routing strategy for LiteLLM using Conduit's contextual bandits.

    This strategy integrates Conduit's machine learning-based model selection
    with LiteLLM's router, enabling intelligent routing across 100+ LLM providers.

    Example:
        >>> from litellm import Router
        >>> from conduit_litellm import ConduitRoutingStrategy
        >>>
        >>> router = Router(model_list=[...])
        >>> router.set_custom_routing_strategy(
        ...     ConduitRoutingStrategy(use_hybrid=True)
        ... )
    """

    def __init__(
        self,
        conduit_router: Optional[Router] = None,
        **conduit_config: Any
    ):
        """Initialize Conduit routing strategy.

        Args:
            conduit_router: Optional pre-configured Conduit router.
            **conduit_config: Additional Conduit configuration options.
                - use_hybrid: Enable UCB1→LinUCB warm start (default: False)
                - embedding_model: Sentence transformer model (default: all-MiniLM-L6-v2)
                - cache_enabled: Enable Redis caching (default: False)
        """
        self.conduit_router = conduit_router
        self.conduit_config = conduit_config
        self._initialized = False
        self._router = None  # Will be set by LiteLLM

    async def _initialize_from_litellm(self, router: Any) -> None:
        """Initialize Conduit router from LiteLLM model list on first call.

        Args:
            router: LiteLLM router instance with model_list.
        """
        if self._initialized:
            return

        # Store LiteLLM router reference
        self._router = router

        # Extract model IDs from LiteLLM model_list
        model_ids = [
            deployment["model_info"]["id"]
            for deployment in router.model_list
        ]

        # Initialize Conduit router if not provided
        if not self.conduit_router:
            self.conduit_router = Router(
                models=model_ids,
                **self.conduit_config
            )

        self._initialized = True

    async def async_get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List[Any]]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Select optimal LiteLLM deployment using Conduit's ML routing.

        This method is called by LiteLLM to select which deployment to use
        for a given request. Conduit analyzes the query and selects the
        optimal model based on learned performance characteristics.

        Args:
            model: Model group name (e.g., "gpt-4").
            messages: Chat messages for the request.
            input: Alternative input format.
            specific_deployment: If True, must return exact deployment.
            request_kwargs: Additional request parameters.

        Returns:
            Selected deployment from litellm.router.model_list.
        """
        # Initialize on first call
        await self._initialize_from_litellm(self._router)

        # Extract query text from messages or input
        query_text = self._extract_query_text(messages, input)

        # Route through Conduit
        query = Query(text=query_text)
        decision = await self.conduit_router.route(query)

        # Find matching deployment in LiteLLM's model_list
        for deployment in self._router.model_list:
            if deployment["model_info"]["id"] == decision.selected_model:
                # TODO: Store routing context for feedback loop (Issue #13)
                return deployment

        # Fallback: return first deployment for this model group
        # (This should rarely happen if model_list is configured correctly)
        for deployment in self._router.model_list:
            if deployment.get("model_name") == model:
                return deployment

        # Last resort: return first deployment
        return self._router.model_list[0]

    def _extract_query_text(
        self,
        messages: Optional[List[Dict[str, str]]],
        input: Optional[Union[str, List[Any]]]
    ) -> str:
        """Extract query text from LiteLLM request format.

        Args:
            messages: Chat messages (OpenAI format).
            input: Alternative input format.

        Returns:
            Extracted query text.
        """
        if messages:
            # Get last user message
            return messages[-1].get("content", "")
        elif isinstance(input, str):
            return input
        elif isinstance(input, list):
            return " ".join(str(x) for x in input)
        return ""

    async def record_feedback(
        self,
        deployment_id: str,
        cost: float,
        latency: float,
        quality_score: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """Record feedback to update Conduit's bandit algorithm.

        This method enables Conduit to learn from LiteLLM request outcomes,
        improving routing decisions over time.

        Args:
            deployment_id: ID of deployment that was used.
            cost: Request cost from LiteLLM.
            latency: Request latency from LiteLLM.
            quality_score: Optional explicit quality rating (0-1).
            error: Optional error message if request failed.

        Note:
            Implementation in Issue #13 (feedback collection and learning).
        """
        # TODO: Implement feedback loop in Issue #13
        pass
