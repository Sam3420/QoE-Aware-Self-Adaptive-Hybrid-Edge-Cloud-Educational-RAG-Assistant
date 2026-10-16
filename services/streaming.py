"""Token streaming hook: a caller (the Streamlit chat) sets a sink, and the LLM client sends answer text to it as it is
generated. Without a sink nothing streams and generation is unchanged.

The sink is a context variable, so it reaches the graph nodes running for that call (LangGraph runs nodes in copies
of the caller's context) and no other session. `sink(None)` means "discard what was streamed so far" (an attempt
failed and the router is trying another route).
"""
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

TokenSink = Callable[[str | None], None]

_sink: ContextVar[TokenSink | None] = ContextVar("token_sink", default=None)


def current_sink() -> TokenSink | None:
    return _sink.get()


@contextmanager
def streaming_to(sink: TokenSink | None) -> Iterator[None]:
    """Send generated answer text to `sink` for the duration of the block."""
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)
