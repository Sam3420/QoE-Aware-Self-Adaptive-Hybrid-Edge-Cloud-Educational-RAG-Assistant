"""Knowledge preparation for a session: topic lesson-note indexes and video transcript indexes.

Video pipeline (architecture §4.4): video → transcript → preprocessing → chunking → embeddings → FAISS.
A video is indexed once under ``yt_<video_id>`` and reused by later sessions.
"""
import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from config import settings
from knowledge_prep.build_index import chunk_text
from services.vector_store import EmbeddingError, IndexNotFoundError
from services.youtube.models import TranscriptStatus
from services.youtube.ranking import LANGUAGE_CODES
from services.youtube.transcripts import clean_text, truncate_record
from telemetry import tracing


logger = logging.getLogger(__name__)

# One lock per index: a background build (prefetch, question-bank warm-up) and the learner's own request never build
# the same index twice or read it half-written; the second caller waits and then reuses the finished index.
_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
_locks_guard = threading.Lock()
_prefetcher = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-prefetch")


def index_lock(resource_id: str) -> threading.Lock:
    with _locks_guard:
        return _locks[resource_id]


class KnowledgeUnavailable(RuntimeError):
    """The resource cannot be turned into a usable index (no transcript, empty text, embedding failure)."""


def video_resource_id(video_id: str) -> str:
    return f"yt_{video_id}"


def index_ready(store: Any, resource_id: str) -> bool:
    """True when the index exists and was built with the current embedder (so it can be searched)."""
    exists = getattr(store, "exists", None)
    if exists is None:  # a search-only backend (test doubles) serves every resource
        return True
    if not exists(resource_id):
        return False
    load = getattr(store, "_load", None)
    if load is None:  # fakes
        return True
    try:
        load(resource_id)
        return True
    except (IndexNotFoundError, EmbeddingError, ValueError):
        return False


def ensure_topic_index(store: Any, topic_id: str) -> bool:
    """Build the topic's lesson-note index if it is missing or stale; return True if it was built now.

    The topic index grounds the topic-level pre/post-test questions. Raises KnowledgeUnavailable when the
    topic has no lesson notes.
    """
    if index_ready(store, topic_id):
        return False
    with index_lock(topic_id):
        if index_ready(store, topic_id):  # built by another thread while we waited
            return False
        path = settings.knowledge_samples_dir / f"{topic_id}.txt"
        if not path.exists():
            raise KnowledgeUnavailable(f"No lesson notes for topic {topic_id!r} ({path.name} is missing).")
        chunks = chunk_text(path.read_text(encoding="utf-8"), resource_id=topic_id, size=settings.chunk_size_words,
                            overlap=settings.chunk_overlap_words, source=path.name)
        store.build(topic_id, chunks)
        return True


def transcript_languages(profile_language: str | None) -> list[str]:
    """The learner's preferred language only, so answers are retrieved from text in that language."""
    language = (profile_language or "").strip().lower()
    code = LANGUAGE_CODES.get(language, language if len(language) == 2 else None)
    return [code] if code else list(settings.transcript_languages)


def index_video(store: Any, transcripts: Any, video: dict[str, Any], languages: list[str]) -> dict[str, Any]:
    """Fetch, clean, chunk and embed one video's transcript; return what was indexed.

    Raises KnowledgeUnavailable with the transcript status when there is nothing usable to index.
    """
    with tracing.observation("fetch-transcript", as_type="tool",
                             input={"video_id": video["video_id"], "languages": list(languages)},
                             metadata={"source": "youtube-transcript-api", "purpose": "index-video"}) as obs:
        record = transcripts.fetch(video["video_id"], languages)
        ok = record.status == TranscriptStatus.OK
        obs.update(output={"status": record.status.value, "language": record.language_code,
                           "caption_type": record.caption_type, "from_cache": record.from_cache, "error": record.error},
                   **({} if ok else {"level": "WARNING", "status_message": record.error or record.status.value}))
    if record.status != TranscriptStatus.OK:
        raise KnowledgeUnavailable(f"transcript {record.status.value}" + (f": {record.error}" if record.error else ""))
    record = truncate_record(record, settings.transcript_max_words)
    text = clean_text(record.text)
    if len(text.split()) < 50:
        raise KnowledgeUnavailable(f"transcript too short to teach from ({len(text.split())} words)")
    resource_id = video_resource_id(video["video_id"])
    chunks = chunk_text(text, resource_id=resource_id, size=settings.chunk_size_words,
                        overlap=settings.chunk_overlap_words, source=video.get("title") or video["video_id"])
    try:
        meta = store.build(resource_id, chunks)
    except EmbeddingError as exc:
        raise KnowledgeUnavailable(f"embedding failed: {exc}") from exc
    return {"resource_id": resource_id, "n_chunks": len(meta.chunks), "embedding_model": meta.embedding_model,
            "caption_language": record.language_code, "caption_type": record.caption_type,
            "truncated": record.truncated, "words": len(text.split())}


def ensure_video_index(store: Any, transcripts: Any, video: dict[str, Any], languages: list[str]) -> dict[str, Any]:
    """The video's index info, building it unless it exists (waits for a background build of the same video)."""
    resource_id = video_resource_id(video["video_id"])
    if index_ready(store, resource_id):
        return {"resource_id": resource_id, "reused_index": True}
    with index_lock(resource_id):
        if index_ready(store, resource_id):  # the prefetch finished it while we waited
            return {"resource_id": resource_id, "reused_index": True}
        return {**index_video(store, transcripts, video, languages), "reused_index": False}


def prefetch_video_indexes(store: Any, transcripts: Any, videos: list[dict[str, Any]], languages: list[str],
                           on_indexed: Callable[[dict[str, Any], dict[str, Any]], None] | None = None) -> int:
    """Index these videos' transcripts in the background while the learner is choosing; returns how many were
    queued. `on_indexed(video, info)` runs after a new index is built. Failures are only logged: prepare_knowledge
    retries the chosen video and reports why it failed."""
    queued = 0
    for video in videos:
        if index_ready(store, video_resource_id(video["video_id"])):
            continue

        def build(video: dict[str, Any] = video) -> None:
            try:
                info = ensure_video_index(store, transcripts, video, languages)
                if on_indexed is not None and not info["reused_index"]:
                    on_indexed(video, info)
            except Exception as exc:  # KnowledgeUnavailable, network errors: the foreground path handles them
                logger.info("prefetch of %s skipped: %s", video["video_id"], exc)

        _prefetcher.submit(tracing.with_current_context(build))
        queued += 1
    return queued

