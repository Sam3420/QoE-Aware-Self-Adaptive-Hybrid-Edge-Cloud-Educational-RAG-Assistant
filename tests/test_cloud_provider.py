"""Cloud route provider: Groq only (no Hugging Face LLM)."""
import httpx
import pytest

from agent.deps import make_cloud_client
from config import Settings
from services.llm_clients import GenerationRequest, LLMError


def _settings(**kw):
    return Settings(_env_file=None, **kw)


def _request():
    return GenerationRequest(system="Be brief.", prompt="Why do shoots bend to light?", max_tokens=200,
                             temperature=0.2, operation="generate-answer")


def test_cloud_route_is_groq():
    client = make_cloud_client(_settings(GROQ_API_KEY="gsk_x", HF_TOKEN="hf_x"))
    assert (client.label, client.model) == ("groq", "openai/gpt-oss-120b") and client.is_available()


def test_misspelt_grok_variable_is_accepted():
    assert make_cloud_client(_settings(GROK_API_KEY="gsk_x")).label == "groq"


def test_without_a_groq_key_the_cloud_route_is_unavailable_even_with_an_hf_token():
    client = make_cloud_client(_settings(HF_TOKEN="hf_x"))
    assert client.label == "groq" and not client.is_available()
    with pytest.raises(LLMError, match="GROQ_API_KEY is not configured"):
        client.generate(_request())


def test_groq_request_carries_reasoning_effort_and_reads_the_answer(monkeypatch):
    sent = {}

    def fake_post(url, json, headers, timeout):
        sent.update(url=url, body=json, auth=headers["Authorization"])
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "choices": [{"message": {"role": "assistant", "content": "Auxin collects on the shady side."},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 12}})

    monkeypatch.setattr(httpx, "post", fake_post)
    result = make_cloud_client(_settings(GROQ_API_KEY="gsk_x")).generate(_request())
    assert result.text == "Auxin collects on the shady side." and result.provider == "cloud"
    assert result.model == "openai/gpt-oss-120b" and result.completion_tokens == 12
    assert sent["url"] == "https://api.groq.com/openai/v1/chat/completions" and sent["auth"] == "Bearer gsk_x"
    assert sent["body"]["reasoning_effort"] == "low" and sent["body"]["max_tokens"] == 200


def test_reasoning_effort_is_only_sent_to_reasoning_models():
    client = make_cloud_client(_settings(GROQ_API_KEY="gsk_x", groq_model="llama-3.1-8b-instant"))
    assert client._extra == {}


def test_groq_errors_and_empty_answers_are_explained(monkeypatch):
    def rate_limited(url, json, headers, timeout):
        return httpx.Response(429, request=httpx.Request("POST", url),
                              json={"error": {"message": "Rate limit reached for model openai/gpt-oss-120b"}})

    client = make_cloud_client(_settings(GROQ_API_KEY="gsk_x"))
    monkeypatch.setattr(httpx, "post", rate_limited)
    with pytest.raises(LLMError, match="groq generation failed.*Rate limit reached"):
        client.generate(_request())

    def thought_too_long(url, json, headers, timeout):
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}]})

    monkeypatch.setattr(httpx, "post", thought_too_long)
    with pytest.raises(LLMError, match="empty answer.*finish_reason=length"):
        client.generate(_request())
