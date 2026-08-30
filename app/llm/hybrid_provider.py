from __future__ import annotations

from app.core.config import Settings
from app.llm.provider import LLMProvider
from app.llm.types import LLMGenerateRequest, LLMGenerateResult, LLMProviderError


class LocalLLMProvider:
    provider_name = "local"

    def __init__(self, *, model_id: str) -> None:
        self.model_id = model_id

    def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        raise LLMProviderError("Local model inference is not available in the current repository configuration.")


class HybridLLMProvider:
    def __init__(
        self,
        *,
        cloud_provider: LLMProvider,
        local_provider: LLMProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.cloud_provider = cloud_provider
        self.local_provider = local_provider
        self.settings = settings or Settings()

    def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        if not getattr(self.settings, "local_llm_enabled", False):
            return self.cloud_provider.generate(request)

        if self.local_provider is None:
            return self.cloud_provider.generate(request)

        try:
            return self.local_provider.generate(request)
        except LLMProviderError:
            return self.cloud_provider.generate(request)
