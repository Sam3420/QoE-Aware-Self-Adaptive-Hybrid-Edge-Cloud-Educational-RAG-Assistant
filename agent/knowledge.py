"""Knowledge preparation for a session: topic lesson-note indexes and video transcript indexes.

Video pipeline (architecture §4.4): video → transcript → preprocessing → chunking → embeddings → FAISS.
A video is indexed once under ``yt_<video_id>`` and reused by later sessions.
"""
from typing import Any

from config import settings
from knowledge_prep.build_index import chunk_text
from services.vector_store import EmbeddingError, IndexNotFoundError
from services.youtube.models import TranscriptStatus
from services.youtube.ranking import LANGUAGE_CODES
from services.youtube.transcripts import clean_text, truncate_record


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
    record = transcripts.fetch(video["video_id"], languages)
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
