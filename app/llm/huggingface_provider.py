import httpx

from app.llm.types import LLMGenerateRequest, LLMGenerateResult, LLMProviderError


class HuggingFaceInferenceProvider:
    provider_name = "huggingface"

    def __init__(
        self,
        *,
        token: str,
        inference_url: str,
        timeout_seconds: float,
    ) -> None:
        if not token:
            raise LLMProviderError("Hugging Face API token is not configured.")
        self._token = token
        self._inference_url = inference_url
        self._timeout_seconds = timeout_seconds

    def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        try:
            response = httpx.post(
                self._inference_url,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": request.model_id,
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                },
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
