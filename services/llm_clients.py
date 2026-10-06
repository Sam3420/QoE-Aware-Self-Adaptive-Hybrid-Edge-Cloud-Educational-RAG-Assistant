"""LLM clients: local SLM via Ollama, cloud LLM via Hugging Face, and a model-free extractive fallback."""
import re
import time
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field, SecretStr

# Small models misread bare language codes (e.g. answer "hi" for Hindi), so prompts use names.
_LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "bn": "Bengali", "te": "Telugu", "ta": "Tamil", "mr": "Marathi",
    "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "or": "Odia", "ur": "Urdu",
}

_COMPETENCY_STYLE = {
    "beginner": "Use very simple words, short sentences and one everyday example.",
    "intermediate": "Explain clearly with the key terms and one example.",
    "advanced": "Be precise and concise; include the underlying mechanism.",
}


class GenerationRequest(BaseModel):
    """Everything a client needs to answer one question."""

    system: str
    prompt: str = Field(min_length=1)
    context_chunks: list[str] = Field(default_factory=list)  # raw evidence, used by the extractive client
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0, le=2.0)
    # Optional JSON schema for structured output; Ollama enforces it during decoding, other clients ignore it.
    json_schema: dict[str, Any] | None = None


class GenerationResult(BaseModel):
    """One generated answer and its cost."""

    text: str
    provider: str
    model: str
    latency_ms: float = Field(ge=0.0)
    completion_tokens: int | None = None


class LLMError(RuntimeError):
    """Raised when a provider cannot produce an answer."""


class LLMClient(Protocol):
    """Common interface of every provider."""

    provider: str
    model: str

    def is_available(self) -> bool: ...

    def generate(self, request: GenerationRequest) -> GenerationResult: ...


def build_request(query: str, chunks: list[str], *, grounded: bool, competency: str, language: str,
                  max_tokens: int, temperature: float) -> GenerationRequest:
    """Build a competency-adapted prompt from the question and chunks.

    grounded=False means retrieval confidence stayed below the threshold, so the model is told
    the lesson probably does not cover the question.
    """
    style = _COMPETENCY_STYLE.get(competency, _COMPETENCY_STYLE["beginner"])
    system = (
        "You are a patient teaching assistant for rural school learners. "
        f"{style} Answer in {_LANGUAGE_NAMES.get(language, language or 'English')}. "
        "Keep the answer under 120 words. "
        "Use only the provided context; if it does not contain the answer, say so briefly."
    )
    # Small models follow instructions in the user turn more reliably than in the system prompt.
    caution = (
        "Note: this question is probably NOT covered by the lesson. Unless the context clearly answers it, "
        "reply in one or two sentences that this lesson does not cover it, and suggest asking about the lesson.\n\n"
        if not grounded else ""
    )
    # A bare topic ("photosynthesis?") reads as a yes/no question to small models; make it explicit.
    if len(query.split()) <= 3:
        query = f"Explain {query.strip().rstrip('?')}."
    if chunks:
        context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
        prompt = f"{caution}Context from the lesson video:\n{context}\n\nQuestion: {query}\nAnswer:"
    else:
        prompt = f"{caution}No lesson context was found.\n\nQuestion: {query}\nAnswer:"
    return GenerationRequest(system=system, prompt=prompt, context_chunks=chunks,
                             max_tokens=max_tokens, temperature=temperature)


class OllamaClient:
    """Local small language model served by Ollama."""

    provider = "local"

    def __init__(self, *, model: str, base_url: str, timeout_s: float, health_timeout_s: float) -> None:
        self.model = model
        self._base = base_url.rstrip("/")
        self._health_timeout = health_timeout_s
        self._http = httpx.Client(timeout=timeout_s)

    def is_available(self) -> bool:
        """True if Ollama is up and the model is pulled."""
        try:
            resp = self._http.get(f"{self._base}/api/tags", timeout=self._health_timeout)
            resp.raise_for_status()
            names = {m.get("name") for m in resp.json().get("models", [])}
            return self.model in names or f"{self.model}:latest" in names
        except (httpx.HTTPError, ValueError):
            return False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Answer with /api/chat (non-streaming)."""
        body = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": request.system},
                         {"role": "user", "content": request.prompt}],
            "options": {"num_predict": request.max_tokens, "temperature": request.temperature},
        }
        if request.json_schema is not None:
            body["format"] = request.json_schema
        start = time.perf_counter()
        try:
            resp = self._http.post(f"{self._base}/api/chat", json=body)
            resp.raise_for_status()
            data = resp.json()
            text = data["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise LLMError(f"Ollama generation failed ({self.model}): {exc}") from exc
        if not text:
            raise LLMError(f"Ollama returned an empty answer ({self.model}).")
        return GenerationResult(text=text, provider=self.provider, model=self.model,
                                latency_ms=(time.perf_counter() - start) * 1000,
                                completion_tokens=data.get("eval_count"))


class HuggingFaceClient:
    """Cloud LLM through the Hugging Face OpenAI-compatible router."""

    provider = "cloud"

    def __init__(self, *, model: str, url: str, token: SecretStr | None, timeout_s: float) -> None:
        self.model = model
        self._url = url
        self._token = token
        self._timeout = timeout_s

    def is_available(self) -> bool:
        """True when an API token is configured (no network probe, to save quota)."""
        return self._token is not None and bool(self._token.get_secret_value())

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Answer with a chat-completions call."""
        if not self.is_available():
            raise LLMError("HF_TOKEN is not configured.")
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": request.system},
                         {"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        headers = {"Authorization": f"Bearer {self._token.get_secret_value()}"}
        start = time.perf_counter()
        try:
            resp = httpx.post(self._url, json=body, headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Hugging Face generation failed ({self.model}): {exc}") from exc
        if not text:
            raise LLMError(f"Hugging Face returned an empty answer ({self.model}).")
        return GenerationResult(text=text, provider=self.provider, model=self.model,
                                latency_ms=(time.perf_counter() - start) * 1000,
                                completion_tokens=(data.get("usage") or {}).get("completion_tokens"))


class ExtractiveClient:
    """Model-free last resort: quotes the leading sentences of the best retrieved chunk."""

    provider = "extractive"
    model = "extractive"

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def is_available(self) -> bool:
        """Always available."""
        return True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return the first sentences of the best chunk, or a polite 'not found'."""
        start = time.perf_counter()
        if request.context_chunks:
            sentences = re.split(r"(?<=[.!?])\s+", request.context_chunks[0].strip())
            text = "From the lesson: " + " ".join(sentences[: self.max_sentences])
        else:
            text = "Sorry, I could not find this in the lesson. Please try asking in a different way."
        return GenerationResult(text=text, provider=self.provider, model=self.model,
                                latency_ms=(time.perf_counter() - start) * 1000)
