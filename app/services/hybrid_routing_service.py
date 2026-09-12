from __future__ import annotations

from statistics import mean
from typing import Any


class HybridRoutingService:
    def __init__(
        self,
        *,
        local_provider_available: bool = True,
        cloud_provider_available: bool = True,
    ) -> None:
        self.local_provider_available = local_provider_available
        self.cloud_provider_available = cloud_provider_available

    def decide_provider(
        self,
        *,
        strategy: str,
        current_runtime_configuration: dict[str, Any] | None = None,
        latency_ms: int | None = None,
        qoe_score: float | None = None,
        recent_qoe_scores: list[float] | None = None,
    ) -> str:
        current_runtime_configuration = current_runtime_configuration or {}
        configured_provider = str(current_runtime_configuration.get("provider", "cloud")).lower()

        if strategy == "cloud_only":
            return "cloud" if self.cloud_provider_available else self._fallback_provider()

        if strategy == "static_hybrid":
            return "local" if self.local_provider_available else "cloud"

        if strategy == "adaptive_hybrid":
            if qoe_score is not None and qoe_score < 60.0:
                return "local" if self.local_provider_available else "cloud"

            if latency_ms is not None and latency_ms > 2000:
                return "local" if self.local_provider_available else "cloud"

            recent_scores = [float(score) for score in recent_qoe_scores or []]
            if recent_scores and mean(recent_scores) < 60.0:
                return "local" if self.local_provider_available else "cloud"

            return "cloud" if self.cloud_provider_available else self._fallback_provider()

        return configured_provider

    def _fallback_provider(self) -> str:
        if self.local_provider_available:
            return "local"
        if self.cloud_provider_available:
            return "cloud"
        return "unavailable"
