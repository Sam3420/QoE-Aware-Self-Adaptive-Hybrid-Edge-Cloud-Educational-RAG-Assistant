"""HTTP client for the FastAPI backend (app/main.py).

The API exposes two routes, and this module wraps exactly those:
  GET  /health
  POST /sessions/{session_id}/interactions/text-question
Every failure is raised as ApiError with a human-readable message; nothing is ever substituted.
"""
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError

from app.domain.models import TextQuestionResponse

DEFAULT_API_URL = os.getenv("QOE_API_URL", "http://127.0.0.1:8000").rstrip("/")
# The backend's own LLM timeout is LLM_TIMEOUT_SECONDS (30 s); leave headroom for DB work.
DEFAULT_QUESTION_TIMEOUT_S = float(os.getenv("QOE_API_TIMEOUT_S", "75"))
HEALTH_TIMEOUT_S = 3.0


class ApiError(RuntimeError):
    """A failed API call, with the HTTP status and any interaction id the backend recorded."""

    def __init__(self, message: str, *, status_code: int | None = None, detail: Any = None,
                 interaction_id: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail
        self.interaction_id = interaction_id


@dataclass(frozen=True)
class HealthResult:
    """Outcome of one real GET /health call."""

    ok: bool
    url: str
    latency_ms: float | None
    status_code: int | None = None
    body: Any = None
    error: str | None = None
    checked_at: float = 0.0


def check_health(base_url: str = DEFAULT_API_URL, timeout_s: float = HEALTH_TIMEOUT_S) -> HealthResult:
    """Call GET /health and report whether the backend answered {"status": "ok"}."""
    url = f"{base_url}/health"
    started = time.perf_counter()
    try:
        response = httpx.get(url, timeout=timeout_s)
    except httpx.ConnectError:
        return HealthResult(False, url, None, error=f"Connection refused at {base_url}. Is uvicorn running?",
                            checked_at=time.time())
    except httpx.TimeoutException:
        return HealthResult(False, url, None, error=f"No response from {base_url} within {timeout_s:.0f} s.",
                            checked_at=time.time())
    except httpx.HTTPError as exc:
        return HealthResult(False, url, None, error=f"{type(exc).__name__}: {exc}", checked_at=time.time())

    latency_ms = (time.perf_counter() - started) * 1000
    try:
        body = response.json()
    except ValueError:
        body = response.text[:200]
    ok = response.status_code == 200 and isinstance(body, dict) and body.get("status") == "ok"
    error = None if ok else f"Unexpected health response: HTTP {response.status_code} {body!r}"
    return HealthResult(ok, url, latency_ms, response.status_code, body, error, time.time())


def ask_text_question(session_id: str, question: str, *, runtime_configuration_id: str | None = None,
                      base_url: str = DEFAULT_API_URL,
                      timeout_s: float = DEFAULT_QUESTION_TIMEOUT_S) -> TextQuestionResponse:
    """POST a question to the assistant and return the validated response."""
    url = f"{base_url}/sessions/{session_id}/interactions/text-question"
    payload: dict[str, Any] = {"question": question}
    if runtime_configuration_id:
        payload["runtime_configuration_id"] = runtime_configuration_id
    try:
        response = httpx.post(url, json=payload, timeout=timeout_s)
    except httpx.ConnectError as exc:
        raise ApiError(f"Backend unavailable: connection refused at {base_url}. "
                       "Start it with `uvicorn app.main:app` from the repository root.") from exc
    except httpx.TimeoutException as exc:
        raise ApiError(f"The backend did not answer within {timeout_s:.0f} s. The LLM provider may be slow "
                       "or overloaded; try again.") from exc
    except httpx.HTTPError as exc:
        raise ApiError(f"Request failed: {type(exc).__name__}: {exc}") from exc

    if response.status_code == 200:
        try:
            return TextQuestionResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise ApiError(f"The backend returned a malformed response: {exc}", status_code=200) from exc
    raise _error_from_response(response)


def _error_from_response(response: httpx.Response) -> ApiError:
    """Translate the API's documented error responses into messages a learner can act on."""
    status = response.status_code
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = response.text[:300] or None

    if status == 404:
        return ApiError(f"Not found: {detail or 'resource does not exist'}", status_code=status, detail=detail)
    if status == 422:
        if isinstance(detail, list):
            msgs = "; ".join(f"{'.'.join(str(p) for p in d.get('loc', []))}: {d.get('msg')}" for d in detail)
        else:
            msgs = str(detail)
        return ApiError(f"The backend rejected the request: {msgs}", status_code=status, detail=detail)
    if status == 503:
        if isinstance(detail, dict):
            return ApiError(f"LLM provider unavailable: {detail.get('message')}", status_code=status,
                            detail=detail, interaction_id=detail.get("interaction_id"))
        hint = (" Hugging Face credentials (HF_TOKEN) are not configured in the backend's environment."
                if detail == "Assistant provider is not configured." else "")
        return ApiError(f"LLM provider unavailable: {str(detail).rstrip('.')}.{hint}", status_code=status,
                        detail=detail)
    if status == 500:
        return ApiError(f"Backend error (HTTP 500): {detail or 'internal server error'}", status_code=status,
                        detail=detail)
    return ApiError(f"Unexpected HTTP {status}: {detail}", status_code=status, detail=detail)
