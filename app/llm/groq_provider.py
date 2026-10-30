from typing import Any

import httpx

from app.llm.types import LLMGenerateRequest, LLMGenerateResult, LLMProviderError


class GroqProvider:
    """Groq's OpenAI-compatible chat-completions API (the only cloud LLM; GROQ_API_KEY)."""

    provider_name = "groq"

    def __init__(
        self,
        *,
        token: str,
        url: str,
        timeout_seconds: float,
        reasoning_effort: str | None = None,
    ) -> None:
        if not token:
            raise LLMProviderError("Groq API key (GROQ_API_KEY) is not configured.")
        self._token = token
        self._url = url
        self._timeout_seconds = timeout_seconds
        self._reasoning_effort = reasoning_effort

    def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        body: dict[str, Any] = {
            "model": request.model_id,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        # gpt-oss models reason before answering; "low" keeps the reasoning (which counts towards max_tokens) short.
        if self._reasoning_effort and request.model_id.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = self._reasoning_effort

        try:
            response = httpx.post(
                self._url,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMProviderError("LLM provider request timed out.") from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError("LLM provider request failed.") from exc

        try:
            payload = response.json()
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMProviderError("LLM provider returned an invalid response.") from exc

        if not isinstance(text, str) or not text.strip():
            raise LLMProviderError("LLM provider returned an empty response.")

        return LLMGenerateResult(
            text=text.strip(),
            model_provider=self.provider_name,
            model_name=request.model_id,
        )


def make_groq_provider(settings: Any) -> GroqProvider:
    """The FastAPI app's cloud LLM from its settings (raises LLMProviderError without a key)."""
    token = settings.groq_api_key.get_secret_value() if settings.groq_api_key else ""
    return GroqProvider(
        token=token,
        url=settings.groq_url,
        timeout_seconds=settings.llm_timeout_seconds,
        reasoning_effort=settings.groq_reasoning_effort,
    )
