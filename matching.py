"""Matching learner questions and videos to a topic's concepts.

Concept ids are those of knowledge_prep/concepts/<topic_id>.json (see tracker.get_concepts). Both functions reuse what
is already loaded: the agent's nomic-embed-text embedder, and for videos the transcript vectors already stored in the
video's FAISS index (yt_<video_id>), so no transcript is embedded again.

Without an embedder (offline fakes) both fall back to whole-word matching of concept names and aliases.
"""
import logging
from typing import Any

import numpy as np

import tracker
from config import settings
from knowledge_prep.concepts import Vocabulary, concept_document, concept_terms, contains_term, normalise

logger = logging.getLogger(__name__)

__all__ = ["CoverageUnknownError", "match_question", "set_embedder", "video_coverage"]


class CoverageUnknownError(LookupError):
    """The video has no transcript index yet (not chosen or prefetched), so its coverage cannot be judged."""


_embedder_override: Any = None
_concept_vectors: dict[tuple[str, str, str], np.ndarray] = {}  # (topic, vocabulary version, embedder) -> vectors
_coverage_cache: dict[tuple[str, str, str], list[str]] = {}  # (video, topic, vocabulary version) -> concept ids


def set_embedder(embedder: Any) -> None:
    """Use this embedder instead of the agent's (tests); None restores the default. Clears the caches."""
    global _embedder_override
    _embedder_override = embedder
    _concept_vectors.clear()
    _coverage_cache.clear()


def match_question(text: str, topic_id: str) -> tuple[str | None, float]:
    """The concept a learner's question is about, and the confidence in [0, 1].

    1. The question names exactly one concept (its name or an alias, as whole words): that concept, with confidence
       MATCH_LEXICAL_CONFIDENCE (0.9).
    2. Otherwise the most similar concept by embedding (among the named ones when several are named). Confidence is
       the cosine similarity, scaled down when the runner-up is within CONCEPT_MATCH_MIN_MARGIN (an ambiguous
       question: "Voluntary Movement" scored reflex_arc 0.636 vs plant_tropisms 0.635).
    Without an embedder, a question naming several concepts gets half the lexical confidence, and one naming none
    returns (None, 0.0).

    Args:
        text: the learner's question as typed.
        topic_id: catalogue topic id; its vocabulary must exist.

    Returns:
        (concept_id, confidence); (None, 0.0) for an empty question. The caller decides whether the confidence is
        high enough to act on (the coaching nodes record it at COACH_MATCH_MIN_CONFIDENCE, 0.6).

    Raises:
        tracker.VocabularyNotFoundError: the topic has no vocabulary.
    """
    vocabulary = tracker._vocabulary(topic_id)
    key = normalise(text or "")
    if not key:
        return None, 0.0
    named = [c.id for c in vocabulary.concepts if any(contains_term(key, t) for t in concept_terms(c))]
    if len(named) == 1:
        return named[0], settings.match_lexical_confidence
    embedder = _embedder()
    if embedder is None:
        return (named[0], round(settings.match_lexical_confidence / 2, 4)) if named else (None, 0.0)
    ids = [c.id for c in vocabulary.concepts]
    sims = _vectors(vocabulary, embedder) @ np.asarray(embedder.embed_query(text), dtype="float32")
    candidates = [ids.index(c) for c in named] if named else list(range(len(ids)))
    ranked = sorted(candidates, key=lambda i: -sims[i])
    best = float(sims[ranked[0]])
    margin = float(sims[ranked[0]] - sims[ranked[1]]) if len(ranked) > 1 else 1.0
    confidence = max(0.0, min(best, 1.0)) * min(1.0, margin / settings.concept_match_min_margin
                                                if settings.concept_match_min_margin else 1.0)
    return ids[ranked[0]], round(confidence, 4)


def video_coverage(video_id: str, topic_id: str) -> list[str]:
    """Concept ids a YouTube video covers, judged from its transcript.

    A concept is covered when a transcript chunk is at least COVERAGE_MIN_SIMILARITY similar to the concept (the
    chunk vectors stored in the video's FAISS index; nothing is re-embedded), or when the transcript names the concept
    (name or alias) at least COVERAGE_MIN_MENTIONS times.

    Args:
        video_id: YouTube video id (the transcript index is ``yt_<video_id>``).
        topic_id: catalogue topic id; its vocabulary must exist.

    Returns:
        Concept ids in vocabulary order (empty when the transcript covers none).

    Raises:
        tracker.VocabularyNotFoundError: the topic has no vocabulary.
        CoverageUnknownError: the video is not indexed yet (no transcript fetched), so coverage is unknown. Callers
            treat this as "unknown", not as "covers nothing" (agent/nodes/coaching.py shows no line on the card).
    """
    vocabulary = tracker._vocabulary(topic_id)
    cache_key = (video_id, topic_id, _version(vocabulary))
    if cache_key in _coverage_cache:
        return list(_coverage_cache[cache_key])
    texts, chunk_vectors = _transcript(f"yt_{video_id}")
    transcript = normalise(" ".join(texts))
    embedder = _embedder() if chunk_vectors is not None else None
    sims = (_vectors(vocabulary, embedder) @ chunk_vectors.T) if embedder is not None else None
    covered = []
    for i, concept in enumerate(vocabulary.concepts):
        mentions = sum(f" {transcript} ".count(f" {t} ") for t in concept_terms(concept))
        similar = sims is not None and float(sims[i].max()) >= settings.coverage_min_similarity
        if similar or mentions >= settings.coverage_min_mentions:
            covered.append(concept.id)
    _coverage_cache[cache_key] = covered
    return list(covered)


# --- internals -----------------------------------------------------------------------------------------------------

def _embedder() -> Any:
    if _embedder_override is not None:
        return _embedder_override
    from agent.deps import get_services

    return getattr(get_services().crag.store, "embedder", None)


def _version(vocabulary: Vocabulary) -> str:
    return f"{vocabulary.generated_at.isoformat()}|{vocabulary.reviewed_at}|{len(vocabulary.concepts)}"


def _vectors(vocabulary: Vocabulary, embedder: Any) -> np.ndarray:
    key = (vocabulary.topic_id, _version(vocabulary), getattr(embedder, "name", type(embedder).__name__))
    if key not in _concept_vectors:
        _concept_vectors[key] = np.asarray(
            embedder.embed_documents([concept_document(c) for c in vocabulary.concepts]), dtype="float32")
    return _concept_vectors[key]


def _transcript(resource_id: str) -> tuple[list[str], np.ndarray | None]:
    """The video's transcript chunks and their stored vectors (None for a store without vectors, e.g. the fake)."""
    from agent.deps import get_services

    store = get_services().crag.store
    exists = getattr(store, "exists", None)
    if exists is None or not exists(resource_id):
        raise CoverageUnknownError(f"No transcript index {resource_id!r} yet.")
    if hasattr(store, "_load"):  # FaissVectorStore: raises EmbeddingError if built with another embedder
        index, meta = store._load(resource_id)
        return [c.text for c in meta.chunks], np.asarray(index.reconstruct_n(0, index.ntotal), dtype="float32")
    chunks = getattr(store, "built", {}).get(resource_id, [])
    return [getattr(c, "text", str(c)) for c in chunks], None
