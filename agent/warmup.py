"""Background warm-up at app start, so the first learner does not pay one-time costs.

1. Load the local LLM (one tiny prompt) and the embedding model; with OLLAMA_KEEP_ALIVE they then stay loaded.
   Measured on the project machine: a cold gemma3:1b answer took ~70 s, the same call warm ~1.4 s.
2. Generate the question bank of every catalogue topic that has none yet (PREWARM_QUESTION_BANKS), one topic at a
   time. A learner who starts that topic meanwhile waits for the same bank instead of generating it twice.

Runs once per process in a daemon thread and never raises: everything here only saves time later.
"""
import logging
import threading
import time
from typing import Any

from agent.knowledge import KnowledgeUnavailable, ensure_topic_index
from config import Settings, settings
from knowledge_prep.catalog import TOPICS

logger = logging.getLogger(__name__)

_started = False
_started_lock = threading.Lock()


def start_background_warmup(services: Any, cfg: Settings = settings) -> threading.Thread | None:
    """Start the warm-up thread once per process; None when disabled or already started."""
    global _started
    with _started_lock:
        if _started or not cfg.warmup_on_start:
            return None
        _started = True
    thread = threading.Thread(target=warm_up, args=(services, cfg), name="warmup", daemon=True)
    thread.start()
    return thread


def warm_up(services: Any, cfg: Settings = settings) -> dict[str, Any]:
    """Run the warm-up steps; returns what was done (for logs and tests)."""
    report: dict[str, Any] = {"llm_s": None, "embedder_s": None, "banks_generated": [], "errors": []}
    for client in getattr(services.router, "clients", {}).values():
        if hasattr(client, "warm_up"):
            try:
                report["llm_s"] = round(client.warm_up(), 2)
            except Exception as exc:
                report["errors"].append(f"llm warm-up: {type(exc).__name__}: {exc}")
    embedder = getattr(getattr(services.crag, "store", None), "embedder", None)
    if embedder is not None and hasattr(embedder, "embed_query"):
        try:
            start = time.perf_counter()
            embedder.embed_query("warm up")
            report["embedder_s"] = round(time.perf_counter() - start, 2)
        except Exception as exc:
            report["errors"].append(f"embedder warm-up: {type(exc).__name__}: {exc}")
    if cfg.prewarm_question_banks:
        for topic in TOPICS:
            try:
                ensure_topic_index(services.crag.store, topic.topic_id)  # the bank's questions come from the notes
                if services.app.ensure_question_bank(topic.topic_id):
                    report["banks_generated"].append(topic.topic_id)
            except KnowledgeUnavailable as exc:
                report["errors"].append(f"{topic.topic_id}: {exc}")
            except Exception as exc:  # no LLM reachable, DB error: the learner's own request will report it
                report["errors"].append(f"{topic.topic_id}: {type(exc).__name__}: {exc}")
    logger.info("warm-up done: %s", report)
    return report
