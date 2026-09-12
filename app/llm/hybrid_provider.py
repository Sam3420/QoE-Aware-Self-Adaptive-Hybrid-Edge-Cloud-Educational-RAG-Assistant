from __future__ import annotations

from app.core.config import Settings
from app.llm.provider import LLMProvider
from app.llm.types import LLMGenerateRequest, LLMGenerateResult, LLMProviderError


class LocalLLMProvider:
    provider_name = "local"

    def __init__(self, *, model_id: str) -> None:
        self.model_id = model_id

    def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        # Phase 7 local inference is represented by a lightweight local fallback
        # that can still provide a deterministic educational response when a
        # true local model host is not yet configured.
        prompt = (request.prompt or "").strip()
        if not prompt:
            raise LLMProviderError("Local LLM provider received an empty prompt.")

        system_prompt = (request.system_prompt or "Provide a concise educational explanation.").strip()
        answer = (
            f"Local fallback response ({self.model_id}):\n"
            f"{system_prompt} \n\n"
            f"Topic: {prompt}\n"
            f"A concise educational explanation is: {prompt} is the subject of this question, and the best response is to explain it clearly, step by step, using simple language and examples when helpful."
        )

        return LLMGenerateResult(
            text=answer,
            model_provider=self.provider_name,
            model_name=self.model_id,
        )


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
        requested_provider = (request.provider or "").strip().lower()

        if requested_provider in {"cloud", "huggingface", "hf"}:
            return self.cloud_provider.generate(request)

        if requested_provider == "local":
            if self.local_provider is None:
                return self.cloud_provider.generate(request)
            try:
                return self.local_provider.generate(request)
            except LLMProviderError:
                return self.cloud_provider.generate(request)

        # Cloud inference remains the default unless local routing is explicitly enabled.
        if not getattr(self.settings, "local_llm_enabled", False):
            return self.cloud_provider.generate(request)

        if self.local_provider is None:
            return self.cloud_provider.generate(request)

        # A local failure falls back to the cloud provider to keep the interaction available.
        try:
            return self.local_provider.generate(request)
        except LLMProviderError:
            return self.cloud_provider.generate(request)
