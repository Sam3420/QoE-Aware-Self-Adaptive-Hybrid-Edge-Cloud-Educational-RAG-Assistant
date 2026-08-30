import httpx
import pytest

from app.llm.huggingface_provider import HuggingFaceInferenceProvider
from app.llm.types import LLMGenerateRequest, LLMProviderError


class FakeResponse:
    def __init__(self, payload: dict, status_error: Exception | None = None) -> None:
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self) -> None:
        if self.status_error is not None:
            raise self.status_error

    def json(self) -> dict:
        return self.payload


def make_request() -> LLMGenerateRequest:
    return LLMGenerateRequest(
        prompt="Explain atoms.",
        system_prompt="Be educational.",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        max_tokens=128,
        temperature=0.2,
    )


def test_huggingface_provider_sends_chat_completion_request(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse(
            {"choices": [{"message": {"content": "Atoms are tiny units of matter."}}]}
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = HuggingFaceInferenceProvider(
        token="secret-token",
        inference_url="https://router.huggingface.co/v1/chat/completions",
        timeout_seconds=10,
    )

    result = provider.generate(make_request())

    assert captured["url"] == "https://router.huggingface.co/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["json"]["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert captured["json"]["messages"][0]["role"] == "system"
    assert captured["json"]["messages"][1]["content"] == "Explain atoms."
    assert captured["timeout"] == 10
    assert result.text == "Atoms are tiny units of matter."
    assert result.model_provider == "huggingface"


def test_huggingface_provider_sanitizes_http_errors(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        request = httpx.Request("POST", url)
        response = httpx.Response(500, request=request)
        return FakeResponse({}, httpx.HTTPStatusError("boom", request=request, response=response))

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = HuggingFaceInferenceProvider(
        token="secret-token",
        inference_url="https://router.huggingface.co/v1/chat/completions",
        timeout_seconds=10,
    )

    with pytest.raises(LLMProviderError, match="request failed"):
        provider.generate(make_request())


def test_huggingface_provider_rejects_malformed_response(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return FakeResponse({"choices": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = HuggingFaceInferenceProvider(
        token="secret-token",
        inference_url="https://router.huggingface.co/v1/chat/completions",
        timeout_seconds=10,
    )

    with pytest.raises(LLMProviderError, match="invalid response"):
        provider.generate(make_request())
