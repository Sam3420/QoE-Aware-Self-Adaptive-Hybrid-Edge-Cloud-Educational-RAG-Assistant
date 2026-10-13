"""Structured logging plus optional Langfuse tracing for the adaptive loop.

Tracing is a no-op unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are configured (and the
`langfuse` package is installed). A tracing failure never interrupts a learning session.
"""
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from config import settings

logger = logging.getLogger("adaptive")
_client: Any = None
_client_checked = False


class _NoopSpan:
    def update(self, **_: Any) -> None: ...

    def score(self, **_: Any) -> None: ...


def _langfuse() -> Any:
    """The Langfuse client, created once, or None when tracing is not configured."""
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        from langfuse import Langfuse

        _client = Langfuse(public_key=settings.langfuse_public_key.get_secret_value(),
                           secret_key=settings.langfuse_secret_key.get_secret_value(),
                           host=settings.langfuse_host or None)
    except Exception as exc:  # missing package or bad configuration: keep running without tracing
        logger.warning("Langfuse tracing disabled: %s", exc)
        _client = None
    return _client


def log_event(event: str, **fields: Any) -> None:
    """One JSON log line per adaptive-loop event (policy decision, reward, storage)."""
    logger.info(json.dumps({"event": event, **fields}, default=str, sort_keys=True))


@contextmanager
def observe(name: str, *, input: Any = None, metadata: dict[str, Any] | None = None) -> Iterator[Any]:
    """A Langfuse span when tracing is configured, otherwise a no-op object with the same methods."""
    client = _langfuse()
    if client is None:
        yield _NoopSpan()
        return
    try:
        cm = client.start_as_current_observation(name=name, as_type="span", input=input, metadata=metadata)
        span = cm.__enter__()
    except Exception as exc:
        logger.warning("Langfuse span %s not started: %s", name, exc)
        yield _NoopSpan()
        return
    try:
        yield span
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception as exc:
            logger.warning("Langfuse span %s not closed: %s", name, exc)
