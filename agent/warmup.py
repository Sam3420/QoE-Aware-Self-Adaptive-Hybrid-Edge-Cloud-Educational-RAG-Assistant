"""Background warm-up at app start, so the first learner does not pay one-time costs.

1. Load the local LLM (one tiny prompt) and the embedding model; with OLLAMA_KEEP_ALIVE they then stay loaded.
   Measured on the project machine: a cold gemma3:1b answer took ~70 s, the same call warm ~1.4 s.
   Then the speech-to-text model (PREWARM_STT): ~5 s to load SraVaani on the GPU, otherwise paid by the first
   spoken question.
   Then, if PREWARM_VIDEO_CACHES_ON_START, the YouTube caches of every topic (search, metadata, caption verdicts);
   by default only a learner's own topic is warmed, while they take its pre-test (warm_videos_in_background).
2. Generate the question bank of every catalogue topic that has none yet (PREWARM_QUESTION_BANKS), one topic at a
   time. A learner who starts that topic meanwhile waits for the same bank instead of generating it twice.

Runs once per process in a daemon thread and never raises: everything here only saves time later.
"""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
    if cfg.prewarm_stt and cfg.stt_enabled:
        from services import speech

        if speech.status()["stt"] is None:
            try:
                report["stt_s"] = round(speech.get_stt().load(), 2)
            except Exception as exc:  # not downloaded, gated, no GPU memory: the first spoken question reports it
                report["errors"].append(f"stt warm-up: {type(exc).__name__}: {exc}")
    if cfg.prewarm_video_caches_on_start:
        report["video_caches"] = {}
        for topic in TOPICS:
            try:
                report["video_caches"][topic.topic_id] = services.video_recommender().warm(topic.topic_id)
            except Exception as exc:  # no API key or quota, YouTube unreachable: recommendations report it later
                report["errors"].append(f"video warm-up {topic.topic_id}: {type(exc).__name__}: {exc}")
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


_video_warmer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-warm")
_video_warmed: dict[tuple[str, str | None], float] = {}
_video_warmed_lock = threading.Lock()
VIDEO_REWARM_S = 600.0


def warm_videos_in_background(services: Any, topic_id: str, student_id: str | None = None,
                              cfg: Settings = settings) -> bool:
    """Fill the YouTube caches for this learner's topic while they take the pre-test (never blocks or raises);
    False when disabled or already done for them in the last VIDEO_REWARM_S."""
    if not cfg.prewarm_video_caches:
        return False
    key, now = (topic_id, student_id), time.monotonic()
    with _video_warmed_lock:
        if now - _video_warmed.get(key, float("-inf")) < VIDEO_REWARM_S:
            return False
        _video_warmed[key] = now

    def run() -> None:
        try:
            logger.info("video caches warmed for %s: %s", topic_id, services.video_recommender().warm(topic_id, student_id))
        except Exception as exc:  # an optimisation only
            logger.info("video cache warm-up for %s skipped: %s", topic_id, exc)

    _video_warmer.submit(run)
    return True
