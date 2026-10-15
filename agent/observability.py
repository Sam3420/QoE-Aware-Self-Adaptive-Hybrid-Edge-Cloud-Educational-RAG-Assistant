"""Structured JSON logging for the adaptive loop; Langfuse tracing lives in telemetry/tracing.py.

`observe` is kept as a thin alias of telemetry.tracing.observation for older call sites.
"""
import json
import logging
from typing import Any

from telemetry.tracing import observation as observe  # noqa: F401  (re-export)

logger = logging.getLogger("adaptive")


def log_event(event: str, **fields: Any) -> None:
    """One JSON log line per adaptive-loop event (policy decision, reward, storage)."""
    logger.info(json.dumps({"event": event, **fields}, default=str, sort_keys=True))
