"""Transcript preprocessing and semantic relevance with the project's existing embedder.

The embedder is the same object the RAG pipeline uses (services.vector_store.Embedder), so no
second model is loaded. Transcripts are chunked into bounded word windows with time ranges; only
chunks (never whole transcripts) are embedded, and nothing is sent to an LLM.
Every evidence item is derived from real transcript text and carries its timestamp.
"""
import re
import time
from collections.abc import Sequence

import numpy as np

from services.vector_store import Embedder
from services.youtube.models import (
    EvidenceExcerpt,
    TranscriptChunk,
    TranscriptEvidence,
    TranscriptRecord,
    VideoCandidate,
)
from services.youtube.transcripts import clean_text

_EXAMPLE = re.compile(r"\b(for example|for instance|e\.g\.|let'?s take|suppose|imagine|consider this|"
                      r"here'?s an example|an example)\b", re.IGNORECASE)
_STEP = re.compile(r"\b(step (?:one|two|three|\d+)|first(?:ly)?,|second(?:ly)?,|third(?:ly)?,|next,|then,|"
                   r"finally,|the first step|the next step)", re.IGNORECASE)
_PREREQ = re.compile(r"\b(before we (?:start|begin)|you should (?:already )?know|recall (?:that|from)|"
                     r"prerequisite|remember (?:that|from)|as we learned)\b", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "what", "how", "why", "for", "on", "with",
         "about", "explain", "explained", "video", "lesson", "class"}


def chunk_transcript(record: TranscriptRecord, *, words_per_chunk: int, overlap: int,
                     max_chunks: int) -> list[TranscriptChunk]:
    """Split a transcript into overlapping word windows with start/end times.

    Word times are interpolated within each caption segment. When there are more windows than
    max_chunks, windows are sampled evenly so the whole video stays represented.
    """
    words: list[tuple[str, float, float]] = []
    for seg in record.segments:
        tokens = clean_text(seg.text).split()
        for i, token in enumerate(tokens):
            t = seg.start + seg.duration * i / max(len(tokens), 1)
            words.append((token, t, seg.start + seg.duration))
    if not words:
        return []
    step = max(words_per_chunk - overlap, 1)
    starts = list(range(0, max(len(words) - overlap, 1), step))
    if len(starts) > max_chunks:
        starts = [starts[round(i * (len(starts) - 1) / (max_chunks - 1))] for i in range(max_chunks)] \
            if max_chunks > 1 else [0]
    chunks = []
    for index, start in enumerate(dict.fromkeys(starts)):
        window = words[start:start + words_per_chunk]
        chunks.append(TranscriptChunk(video_id=record.video_id, index=index, start=round(window[0][1], 2),
                                      end=round(window[-1][2], 2), text=" ".join(w for w, _, _ in window),
                                      language_code=record.language_code))
    return chunks


def metadata_text(candidate: VideoCandidate, max_description_chars: int = 600) -> str:
    """Title, tags and the start of the description: what search metadata says the video is about."""
    parts = [candidate.title, " ".join(candidate.tags[:15]), candidate.description[:max_description_chars]]
    return clean_text(" ".join(p for p in parts if p))


def keyword_overlap(objective: str, text: str) -> float:
    """Lexical fallback when embeddings are unavailable: share of objective keywords present in text."""
    keys = {w for w in _WORD.findall(objective.lower()) if w not in _STOP and len(w) > 2}
    if not keys:
        return 0.0
    present = set(_WORD.findall(text.lower()))
    return len(keys & present) / len(keys)


def _merge_seconds(intervals: list[tuple[float, float]]) -> float:
    """Total length of the union of time intervals (overlapping chunks are counted once)."""
    total, current = 0.0, None
    for start, end in sorted(intervals):
        if current is None or start > current[1]:
            if current:
                total += current[1] - current[0]
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    if current:
        total += current[1] - current[0]
    return round(total, 1)


class RelevanceScorer:
    """Embeds texts with the shared embedder and turns cosine similarity into [0, 1] scores."""

    def __init__(self, embedder: Embedder, *, floor: float, ceil: float, top_chunks: int,
                 concept_threshold: float) -> None:
        self.embedder = embedder
        self.floor, self.ceil = floor, ceil
        self.top_chunks = top_chunks
        self.concept_threshold = concept_threshold
        self.embed_ms = 0.0
        self.embeddings = 0

    def normalise(self, sims: np.ndarray) -> np.ndarray:
        """Map cosine similarity to [0, 1] with the calibrated floor/ceiling."""
        return np.clip((np.asarray(sims, dtype=float) - self.floor) / (self.ceil - self.floor), 0.0, 1.0)

    def embed_query(self, text: str) -> np.ndarray:
        started = time.perf_counter()
        vec = self.embedder.embed_query(text)
        self.embed_ms += (time.perf_counter() - started) * 1000
        self.embeddings += 1
        return vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0))
        started = time.perf_counter()
        vecs = self.embedder.embed_documents(list(texts))
        self.embed_ms += (time.perf_counter() - started) * 1000
        self.embeddings += len(texts)
        return vecs

    def topic_similarity(self, objective_vec: np.ndarray, candidates: Sequence[VideoCandidate]) -> dict[str, float]:
        """Normalised similarity between the objective and each video's title/tags/description."""
        if not candidates:
            return {}
        vecs = self.embed_documents([metadata_text(c) for c in candidates])
        sims = self.normalise(vecs @ objective_vec)
        return {c.video_id: round(float(s), 4) for c, s in zip(candidates, sims)}

    def transcript_evidence(self, objective_vec: np.ndarray, chunks: Sequence[TranscriptChunk],
                            chunk_vecs: np.ndarray, concepts: dict[str, np.ndarray]) -> TranscriptEvidence:
        """Score one video's chunks against the objective and gap concepts; collect grounded evidence."""
        if not chunks:
            return TranscriptEvidence(relevance=0.0, concepts_missing=list(concepts), n_chunks=0)
        sims = self.normalise(chunk_vecs @ objective_vec)
        order = np.argsort(-sims, kind="stable")
        relevance = float(np.mean(sims[order[:self.top_chunks]]))
        excerpts = [EvidenceExcerpt(start=chunks[i].start, end=chunks[i].end, similarity=round(float(sims[i]), 4),
                                    text=chunks[i].text[:280] + ("…" if len(chunks[i].text) > 280 else ""))
                    for i in order[:3] if sims[i] > 0]

        covered, missing = {}, []
        for concept, cvec in concepts.items():
            csims = self.normalise(chunk_vecs @ cvec)
            hits = [i for i in range(len(chunks)) if csims[i] >= self.concept_threshold]
            if hits:
                covered[concept] = min(chunks[i].start for i in hits)
            else:
                missing.append(concept)

        relevant = [i for i in range(len(chunks)) if sims[i] >= self.concept_threshold]
        cue_chunks = [chunks[i] for i in relevant]
        return TranscriptEvidence(
            relevance=round(relevance, 4),
            excerpts=excerpts,
            concepts_covered=covered,
            concepts_missing=missing,
            example_timestamps=[c.start for c in cue_chunks if _EXAMPLE.search(c.text)][:5],
            step_timestamps=[c.start for c in cue_chunks if _STEP.search(c.text)][:5],
            prerequisite_timestamps=[c.start for c in chunks if _PREREQ.search(c.text)][:5],
            on_topic_seconds=_merge_seconds([(chunks[i].start, chunks[i].end) for i in relevant]),
            n_chunks=len(chunks),
        )
