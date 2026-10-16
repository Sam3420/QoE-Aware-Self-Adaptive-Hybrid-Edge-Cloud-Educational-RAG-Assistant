"""LLM clients: local SLM via Ollama, cloud LLM via Hugging Face, and a model-free extractive fallback."""
import json
import re
import time
from typing import Any, Protocol

import httpx

from services.streaming import current_sink
from telemetry import tracing
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
    operation: str = "generate-response"  # tracing name of the call (what it does, not which model)

    def messages(self) -> list[dict[str, str]]:
        """The chat messages sent to the model (OpenAI format)."""
        return [{"role": "system", "content": self.system}, {"role": "user", "content": self.prompt}]


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
                             max_tokens=max_tokens, temperature=temperature, operation="generate-answer")


class OllamaClient:
    """Local small language model served by Ollama."""

    provider = "local"

    def __init__(self, *, model: str, base_url: str, timeout_s: float, health_timeout_s: float,
                 keep_alive: str | None = None) -> None:
        self.model = model
        self._base = base_url.rstrip("/")
        self._health_timeout = health_timeout_s
        self._keep_alive = keep_alive  # how long Ollama keeps the model loaded after a call (None: its 5 min default)
        self._http = httpx.Client(timeout=timeout_s)

    def warm_up(self) -> float:
        """Load the model and run one tiny prompt, so the learner's first question does not pay the cold start
        (measured: ~70 s cold vs ~1.4 s warm for gemma3:1b). Returns the seconds it took."""
        start = time.perf_counter()
        body = {"model": self.model, "stream": False, "messages": [{"role": "user", "content": "Hi"}],
                "options": {"num_predict": 1}}
        if self._keep_alive is not None:
            body["keep_alive"] = self._keep_alive
        self._http.post(f"{self._base}/api/chat", json=body).raise_for_status()
        return time.perf_counter() - start

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
        """Answer with /api/chat; streamed to the caller's token sink when one is set (plain-text answers only)."""
        sink = current_sink() if request.json_schema is None else None
        body = {
            "model": self.model,
            "stream": sink is not None,
            "messages": request.messages(),
            "options": {"num_predict": request.max_tokens, "temperature": request.temperature},
        }
        if self._keep_alive is not None:
            body["keep_alive"] = self._keep_alive
        if request.json_schema is not None:
            body["format"] = request.json_schema
        with tracing.observation(request.operation, as_type="generation", input=request.messages(), model=self.model,
                                 model_parameters={"max_tokens": request.max_tokens,
                                                   "temperature": request.temperature,
                                                   "structured_output": request.json_schema is not None},
                                 metadata={"provider": "ollama", "route": self.provider}) as gen:
            start = time.perf_counter()
            try:
                if sink is None:
                    resp = self._http.post(f"{self._base}/api/chat", json=body)
                    resp.raise_for_status()
                    data = resp.json()
                    text = data["message"]["content"].strip()
                else:
                    data, text = self._stream(body, sink)
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                if sink is not None:
                    sink(None)  # drop the partial text: the router may answer with another route
                raise LLMError(f"Ollama generation failed ({self.model}): {exc}") from exc
            if not text:
                if sink is not None:
                    sink(None)
                raise LLMError(f"Ollama returned an empty answer ({self.model}).")
            gen.update(output=text, usage_details=_usage(data.get("prompt_eval_count"), data.get("eval_count")))
            return GenerationResult(text=text, provider=self.provider, model=self.model,
                                    latency_ms=(time.perf_counter() - start) * 1000,
                                    completion_tokens=data.get("eval_count"))


    def _stream(self, body: dict[str, Any], sink: Any) -> tuple[dict[str, Any], str]:
        """POST a streaming chat and pass each piece of text to `sink`; return the final chunk and the full text."""
        parts: list[str] = []
        last: dict[str, Any] = {}
        with self._http.stream("POST", f"{self._base}/api/chat", json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                last = json.loads(line)
                if last.get("error"):
                    raise ValueError(last["error"])
                piece = (last.get("message") or {}).get("content") or ""
                if piece:
                    parts.append(piece)
                    sink(piece)
                if last.get("done"):
                    break
        return last, "".join(parts).strip()


class OpenAICompatibleClient:
    """Cloud LLM behind an OpenAI chat-completions endpoint (Groq, Hugging Face router)."""

    provider = "cloud"

    def __init__(self, *, model: str, url: str, token: SecretStr | None, timeout_s: float,
                 label: str = "huggingface", token_env: str = "HF_TOKEN",
                 extra_body: dict[str, Any] | None = None) -> None:
        self.model = model
        self.label = label  # which service, for traces and error messages
        self._url = url
        self._token = token
        self._token_env = token_env
        self._timeout = timeout_s
        self._extra = dict(extra_body or {})  # provider-specific request fields, e.g. Groq reasoning_effort

    def is_available(self) -> bool:
        """True when an API token is configured (no network probe, to save quota)."""
        return self._token is not None and bool(self._token.get_secret_value())

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Answer with a chat-completions call."""
        if not self.is_available():
            raise LLMError(f"{self._token_env} is not configured.")
        body = {
            "model": self.model,
            "messages": request.messages(),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            **self._extra,
        }
        headers = {"Authorization": f"Bearer {self._token.get_secret_value()}"}  # never traced
        with tracing.observation(request.operation, as_type="generation", input=request.messages(), model=self.model,
                                 model_parameters={"max_tokens": request.max_tokens,
                                                   "temperature": request.temperature},
                                 metadata={"provider": self.label, "route": self.provider}) as gen:
            start = time.perf_counter()
            try:
                resp = httpx.post(self._url, json=body, headers=headers, timeout=self._timeout)
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                raise LLMError(f"{self.label} generation failed ({self.model}): {_http_detail(exc)}") from exc
            if not text:
                # Reasoning models can spend the whole token budget thinking and return no answer text.
                raise LLMError(f"{self.label} returned an empty answer ({self.model}, "
                               f"finish_reason={data['choices'][0].get('finish_reason')}).")
            usage = data.get("usage") or {}
            gen.update(output=text, usage_details=_usage(usage.get("prompt_tokens"), usage.get("completion_tokens")))
            return GenerationResult(text=text, provider=self.provider, model=self.model,
                                    latency_ms=(time.perf_counter() - start) * 1000,
                                    completion_tokens=usage.get("completion_tokens"))


class HuggingFaceClient(OpenAICompatibleClient):
    """The Hugging Face router (kept for existing callers)."""


def _http_detail(exc: Exception) -> str:
    """The error plus the API's own message for HTTP errors (e.g. Groq's rate-limit or model errors)."""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            message = response.json().get("error", {}).get("message", "")
        except ValueError:
            message = response.text[:200]
        return f"HTTP {response.status_code}: {message or response.reason_phrase}"[:500]
    return str(exc)


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
        with tracing.observation("extract-answer", as_type="tool", input={"chunks": len(request.context_chunks)},
                                 metadata={"route": self.provider, "reason": "model-free fallback"}) as obs:
            start = time.perf_counter()
            if request.context_chunks:
                sentences = re.split(r"(?<=[.!?])\s+", request.context_chunks[0].strip())
                text = "From the lesson: " + " ".join(sentences[: self.max_sentences])
            else:
                text = "Sorry, I could not find this in the lesson. Please try asking in a different way."
            obs.update(output=text)
            return GenerationResult(text=text, provider=self.provider, model=self.model,
                                    latency_ms=(time.perf_counter() - start) * 1000)


def _usage(input_tokens: Any, output_tokens: Any) -> dict[str, int] | None:
    """Token usage for Langfuse (enables cost calculation); None when the provider did not report it."""
    usage = {k: int(v) for k, v in (("input", input_tokens), ("output", output_tokens)) if isinstance(v, int)}
    return usage or None
