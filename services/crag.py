"""Corrective RAG (CRAG) retrieval: search, grade chunks, rewrite queries, merge attempts.

Whether to retry is decided by the agent (agent.routing.crag_ok); this service only does the work.
"""
import re
from typing import Protocol

from pydantic import BaseModel, Field

from services.vector_store import RetrievedChunk

# Words dropped from a vague question before it is rewritten.
_FILLER = {
    "what", "is", "are", "the", "a", "an", "of", "to", "in", "how", "why", "does", "do", "can",
    "you", "me", "tell", "about", "explain", "please", "i", "want", "know",
}
# Expansion terms added per corrective attempt (attempt 1 uses the first entry, and so on).
_EXPANSIONS = [
    "definition explanation key concepts",
    "process steps example evidence",
    "summary overview",
]


class SearchBackend(Protocol):
    """What CRAG needs from a vector store."""

    def search(self, resource_id: str, query: str, top_k: int) -> list[RetrievedChunk]: ...

    def score_chunks(self, resource_id: str, query: str, chunk_ids: list[str]) -> dict[str, float]: ...


class RetrievalResult(BaseModel):
    """Outcome of one retrieval attempt."""

    query: str
    chunks: list[RetrievedChunk]  # only chunks graded as relevant
    top_score: float = Field(ge=0.0, le=1.0)  # best raw score, before grading
    n_discarded: int = Field(ge=0)


class CragRetriever:
    """Runs retrieval and corrective retrieval over a vector store."""

    def __init__(self, *, store: SearchBackend, min_chunk_score: float) -> None:
        self.store = store
        self.min_chunk_score = min_chunk_score

    def retrieve(self, resource_id: str, query: str, top_k: int, grade_query: str | None = None) -> RetrievalResult:
        """Search with `query`, grade hits against `grade_query` (the original question), drop irrelevant ones.

        Grading against the original question stops a rewritten query from inflating scores:
        anchoring an off-topic question to the topic finds on-topic chunks, but they still
        don't answer what was asked.
        """
        hits = self.store.search(resource_id, query, top_k)
        if grade_query and grade_query != query and hits:
            graded = self.store.score_chunks(resource_id, grade_query, [h.chunk_id for h in hits])
            hits = sorted((h.model_copy(update={"score": graded.get(h.chunk_id, 0.0)}) for h in hits),
                          key=lambda h: h.score, reverse=True)
        kept = [h for h in hits if h.score >= self.min_chunk_score]
        return RetrievalResult(
            query=query,
            chunks=kept,
            top_score=max((h.score for h in hits), default=0.0),
            n_discarded=len(hits) - len(kept),
        )

    @staticmethod
    def rewrite_query(query: str, topic_id: str, attempt: int) -> str:
        """Rewrite a query for corrective retrieval: keep its keywords, anchor to the topic, expand."""
        keywords = [w for w in re.findall(r"[\w'-]+", query.lower()) if w not in _FILLER]
        topic = topic_id.replace("_", " ").replace("-", " ").strip()
        core = " ".join(keywords) or query.strip()
        expansion = _EXPANSIONS[min(max(attempt, 1), len(_EXPANSIONS)) - 1]
        prefix = f"{topic}: " if topic and topic not in core else ""
        return f"{prefix}{core} {expansion}".strip()

    @staticmethod
    def merge(previous: list[RetrievedChunk], new: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Union chunks across attempts, keeping each chunk's best score, and return the top_k."""
        best: dict[str, RetrievedChunk] = {}
        for chunk in [*previous, *new]:
            if chunk.chunk_id not in best or chunk.score > best[chunk.chunk_id].score:
                best[chunk.chunk_id] = chunk
        return sorted(best.values(), key=lambda c: c.score, reverse=True)[:top_k]
