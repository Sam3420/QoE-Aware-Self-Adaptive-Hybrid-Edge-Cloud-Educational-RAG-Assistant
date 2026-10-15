"""Langfuse tracing (Python SDK v5) behind a small, failure-safe API.

Trace model (see docs/observability.md):
- one trace per unit of work: each learning-session step (start, pre-test, one question, post-test)
  and each video-recommendation request. Steps of one learning session share `session_id` (the
  learning-session id) and `user_id` (the learner id);
- inside a trace, observations use the most specific type: `agent` roots, `generation` for LLM calls
  (model, parameters, token usage), `embedding`, `retriever`, `tool` (YouTube API, transcripts, probes),
  `evaluator` and `chain`/`span` for the remaining steps;
- learner feedback and session outcomes are recorded as scores named after the observed signal.

Everything is a no-op unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set, and no tracing
failure ever propagates into the application. Export-stage masking (`mask_otel_spans`) redacts emails,
phone numbers and Aadhaar-like numbers in every exported attribute, and replaces learner content when
LANGFUSE_CAPTURE_CONTENT=false.
"""
import contextvars
import json
import logging
import re
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from config import settings

logger = logging.getLogger(__name__)
T = TypeVar("T")

INPUT_KEY = "langfuse.observation.input"
OUTPUT_KEY = "langfuse.observation.output"
TEXT_KEYS = (INPUT_KEY, OUTPUT_KEY, "langfuse.observation.metadata", "langfuse.trace.metadata",
             "langfuse.observation.status_message")
CONTENT_HIDDEN = json.dumps("[content capture disabled]")

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<![\w.-])(?:\+91[\s-]?)?[6-9]\d{9}(?![\w.-])")  # Indian mobile numbers
_AADHAAR = re.compile(r"(?<![\w-])\d{4}[ -]?\d{4}[ -]?\d{4}(?![\w-])")  # 12-digit national id

_client: Any = None
_configured = False
_lock = threading.Lock()


def redact(text: str) -> str:
    """Replace emails, phone numbers and Aadhaar-like numbers."""
    text = _EMAIL.sub("[EMAIL]", text)
    text = _AADHAAR.sub("[ID-NUMBER]", text)
    return _PHONE.sub("[PHONE]", text)


def mask_otel_spans(*, params: Any) -> Any:
    """Export-stage masking for every span this client exports (Langfuse `mask_otel_spans` hook).

    Deterministic and fast; it raises only on a programming error, in which case Langfuse drops the
    batch rather than exporting unmasked data.
    """
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches = {}
    for identifier, span in params.spans.items():
        changes: dict[str, Any] = {}
        for key in TEXT_KEYS:
            value = span.attributes.get(key)
            if not isinstance(value, str):
                continue
            new = CONTENT_HIDDEN if (not settings.langfuse_capture_content and key in (INPUT_KEY, OUTPUT_KEY)) \
                else redact(value)
            if new != value:
                changes[key] = new
        if changes:
            patches[identifier] = OtelSpanPatch(set_attributes=changes)
    return MaskOtelSpansResult(span_patches=patches)


def get_client() -> Any:
    """The configured Langfuse client, or None when tracing is not configured (created once)."""
    global _client, _configured
    if _configured:
        return _client
    with _lock:
        if _configured:
            return _client
        _configured = True
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            return None
        try:
            from langfuse import Langfuse

            _client = Langfuse(
                public_key=settings.langfuse_public_key.get_secret_value(),
                secret_key=settings.langfuse_secret_key.get_secret_value(),
                base_url=settings.langfuse_base_url or settings.langfuse_host or None,
                environment=settings.langfuse_environment,
                release=settings.langfuse_release,
                sample_rate=settings.langfuse_sample_rate,
                mask_otel_spans=mask_otel_spans,
            )
        except Exception as exc:  # missing package or bad configuration: run without tracing
            logger.warning("Langfuse tracing disabled: %s", exc)
            _client = None
        return _client


def set_client(client: Any) -> None:
    """Install a client (tests: a Langfuse instance with an in-memory exporter) or None to disable."""
    global _client, _configured
    with _lock:
        _client, _configured = client, True


def reset() -> None:
    """Forget the client so the next call re-reads the settings."""
    global _client, _configured
    with _lock:
        _client, _configured = None, False


def enabled() -> bool:
    return get_client() is not None


class _NoopObservation:
    """Stands in for a Langfuse observation when tracing is off."""

    id = None
    trace_id = None

    def update(self, **_: Any) -> "_NoopObservation":
        return self

    def score(self, **_: Any) -> None: ...

    def score_trace(self, **_: Any) -> None: ...


NOOP = _NoopObservation()


@contextmanager
def observation(name: str, *, as_type: str = "span", input: Any = None, metadata: Any = None,
                **attributes: Any) -> Iterator[Any]:
    """Start an observation as the current one (children nest under it).

    Extra keyword arguments are passed to `start_as_current_observation` (e.g. model, model_parameters,
    usage_details). An exception escaping the block marks the observation ERROR with its message and
    is re-raised unchanged.
    """
    client = get_client()
    if client is None:
        yield NOOP
        return
    try:
        cm = client.start_as_current_observation(name=name, as_type=as_type, input=input, metadata=metadata,
                                                 **attributes)
        obs = cm.__enter__()
    except Exception as exc:
        logger.warning("Langfuse observation %s not started: %s", name, exc)
        yield NOOP
        return
    exc_info: tuple[Any, Any, Any] = (None, None, None)
    try:
        yield obs
    except BaseException as exc:
        exc_info = sys.exc_info()
        if isinstance(exc, Exception):
            _safe(obs.update, level="ERROR", status_message=f"{type(exc).__name__}: {exc}"[:1000])
        raise
    finally:
        try:
            cm.__exit__(*exc_info)
        except Exception as exit_exc:
            logger.warning("Langfuse observation %s not closed: %s", name, exit_exc)


@contextmanager
def trace(name: str, *, user_id: str | None, session_id: str | None, tags: list[str], input: Any = None,
          metadata: dict[str, Any] | None = None, as_type: str = "agent") -> Iterator[Any]:
    """Root observation of one unit of work, with user/session/tags propagated to every child."""
    client = get_client()
    if client is None:
        yield NOOP
        return
    if _inside_trace():  # already inside a trace: nest, keeping that trace's name/session
        with observation(name, as_type=as_type, input=input, metadata=metadata) as obs:
            yield obs
        return
    try:
        from langfuse import propagate_attributes

        attrs = propagate_attributes(
            user_id=user_id, session_id=session_id, tags=tags, trace_name=name,
            metadata={k: str(v)[:200] for k, v in (metadata or {}).items() if v is not None})
        attrs.__enter__()
    except Exception as exc:
        logger.warning("Langfuse trace attributes not set for %s: %s", name, exc)
        attrs = None
    exc_info: tuple[Any, Any, Any] = (None, None, None)
    try:
        with observation(name, as_type=as_type, input=input) as root:
            yield root
    except BaseException:
        exc_info = sys.exc_info()
        raise
    finally:
        if attrs is not None:
            try:
                attrs.__exit__(*exc_info)
            except Exception as exc:
                logger.warning("Langfuse trace attributes not closed for %s: %s", name, exc)


def _inside_trace() -> bool:
    """True when an OpenTelemetry span is active in this context (checked silently, unlike the client)."""
    try:
        from opentelemetry import trace as otel_trace

        return otel_trace.get_current_span().get_span_context().is_valid
    except Exception:
        return False


def current_trace_id() -> str | None:
    client = get_client()
    return _safe(client.get_current_trace_id) if client is not None else None


def trace_url(trace_id: str | None) -> str | None:
    client = get_client()
    return _safe(client.get_trace_url, trace_id=trace_id) if client is not None and trace_id else None


def score(*, name: str, value: float | str, trace_id: str | None = None, session_id: str | None = None,
          data_type: str | None = None, comment: str | None = None, metadata: dict[str, Any] | None = None) -> None:
    """Record a score on a trace or a session (no-op without tracing; never raises)."""
    client = get_client()
    if client is None or (trace_id is None and session_id is None) or value is None:
        return
    _safe(client.create_score, name=name, value=value, trace_id=trace_id, session_id=session_id,
          data_type=data_type, comment=redact(comment) if comment else None, metadata=metadata)


def with_current_context(fn: Callable[..., T]) -> Callable[..., T]:
    """Bind fn to a copy of the caller's context, so observations created in a worker thread nest
    under the caller's observation instead of starting an orphan trace (thread pools do not inherit
    OpenTelemetry context)."""
    ctx = contextvars.copy_context()

    def run(*args: Any, **kwargs: Any) -> T:
        return ctx.run(fn, *args, **kwargs)

    return run


def flush() -> None:
    """Send buffered observations now (use in short-lived scripts)."""
    client = get_client()
    if client is not None:
        _safe(client.flush)


def shutdown() -> None:
    """Flush and stop background threads (end of scripts / service shutdown)."""
    client = get_client()
    if client is not None:
        _safe(client.shutdown)


def _safe(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T | None:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # tracing must never break the application
        logger.warning("Langfuse call %s failed: %s", getattr(fn, "__name__", fn), exc)
        return None


def preview(text: Any, limit: int = 300) -> Any:
    """Shorten long text for observation payloads."""
    if isinstance(text, str) and len(text) > limit:
        return text[:limit] + f"… [{len(text)} chars]"
    return text
