"""LLM router: registry of providers with cached health checks and a uniform generate call.

Which provider to use is decided by the agent (the llm_router node, following the policy's route);
this service reports availability and executes the call.
"""
import time

from services.llm_clients import GenerationRequest, GenerationResult, LLMClient, LLMError


class LLMRouter:
    """Holds the local, cloud and extractive clients, keyed by provider name."""

    def __init__(self, clients: list[LLMClient], *, health_ttl_s: float = 30.0) -> None:
        self.clients: dict[str, LLMClient] = {c.provider: c for c in clients}
        self.health_ttl_s = health_ttl_s
        self._health: dict[str, tuple[float, bool]] = {}

    def is_available(self, provider: str) -> bool:
        """Return the provider's health, re-probing at most once per TTL."""
        client = self.clients.get(provider)
        if client is None:
            return False
        checked_at, ok = self._health.get(provider, (float("-inf"), False))
        if time.monotonic() - checked_at > self.health_ttl_s:
            ok = client.is_available()
            self._health[provider] = (time.monotonic(), ok)
        return ok

    def model_name(self, provider: str) -> str:
        """Return the model id behind a provider."""
        return self.clients[provider].model

    def generate(self, provider: str, request: GenerationRequest) -> GenerationResult:
        """Generate with one provider; mark it unhealthy if the call fails."""
        client = self.clients.get(provider)
        if client is None:
            raise LLMError(f"Unknown LLM provider '{provider}'.")
        try:
            return client.generate(request)
        except LLMError:
            self._health[provider] = (time.monotonic(), False)
            raise
