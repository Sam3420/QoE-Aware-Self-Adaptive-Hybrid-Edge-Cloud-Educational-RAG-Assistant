"""Concept vocabulary per topic: the 6-12 concepts a chapter is tracked by (knowledge_prep/concepts/<topic_id>.json).

A vocabulary is drafted by an LLM from the topic's lesson notes (scripts/extract_concepts.py) and stays
``"status": "draft"`` until a person reviews it and marks it ``"approved"``. Every concept carries an ``evidence``
sentence that must appear verbatim in the notes, so a concept the notes do not teach is rejected automatically.

ConceptMatcher maps a free-text concept label (e.g. a question-bank item's ``concept``) to a vocabulary concept:
exact name/id/alias first, then embedding similarity, else no match.
"""
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from config import PROJECT_ROOT, settings

CONCEPTS_DIR = PROJECT_ROOT / "knowledge_prep" / "concepts"
MIN_CONCEPTS, MAX_CONCEPTS = 6, 12
_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class Concept(BaseModel):
    id: str  # snake_case, stable: tracker rows reference it
    name: str = Field(min_length=2, max_length=80)
    description: str = Field(min_length=10, max_length=300)
    aliases: list[str] = Field(default_factory=list)  # other names learners or questions use
    evidence: str = Field(min_length=10)  # a sentence from the lesson notes that teaches the concept
    # Question-bank positions that test this concept (derived from question_concept_map by concepts/extract.py).
    source_question_ids: list[int] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _snake_case(cls, value: str) -> str:
        if not _ID.match(value):
            raise ValueError(f"concept id {value!r} must be snake_case (a-z, 0-9, _)")
        return value


class QuestionMapping(BaseModel):
    """The concept one question-bank item tests (keyed in question_concept_map by its bank position)."""

    concept_id: str
    question: str  # the item's text when mapped: the backfill checks the bank position still holds this question
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool = False
    note: str = ""  # why it needs review (low confidence, disagrees with the item's own concept label, ...)


class Vocabulary(BaseModel):
    topic_id: str
    status: Literal["draft", "approved"] = "draft"
    generated_by: str  # e.g. "groq:openai/gpt-oss-120b"
    generated_at: datetime
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    concepts: list[Concept]
    # Bank position (as a string) -> concept, written by concepts/extract.py; kept by every load/save of this file.
    question_concept_map: dict[str, QuestionMapping] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _well_formed(self) -> "Vocabulary":
        if not MIN_CONCEPTS <= len(self.concepts) <= MAX_CONCEPTS:
            raise ValueError(f"expected {MIN_CONCEPTS}-{MAX_CONCEPTS} concepts, got {len(self.concepts)}")
        for field in ("id", "name"):
            values = [_key(getattr(c, field)) for c in self.concepts]
            if len(set(values)) != len(values):
                raise ValueError(f"concept {field}s must be unique")
        if self.status == "approved" and not self.reviewed_by:
            raise ValueError("an approved vocabulary needs reviewed_by")
        ids = {c.id for c in self.concepts}
        unknown = sorted({m.concept_id for m in self.question_concept_map.values()} - ids)
        if unknown:
            raise ValueError(f"question_concept_map uses unknown concept ids: {unknown}")
        return self

    def get(self, concept_id: str) -> Concept | None:
        return next((c for c in self.concepts if c.id == concept_id), None)


def path_for(topic_id: str, directory: Path | None = None) -> Path:
    return (directory or CONCEPTS_DIR) / f"{topic_id}.json"


def load_vocabulary(topic_id: str, directory: Path | None = None) -> Vocabulary | None:
    """The topic's vocabulary, or None if none has been drafted yet."""
    path = path_for(topic_id, directory)
    return Vocabulary.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None


def save_vocabulary(vocabulary: Vocabulary, directory: Path | None = None) -> Path:
    directory = directory or CONCEPTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = path_for(vocabulary.topic_id, directory)
    path.write_text(vocabulary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def evidence_problems(vocabulary: Vocabulary, notes: str) -> list[str]:
    """Concepts whose evidence sentence is not in the notes (compared ignoring case, spacing and punctuation)."""
    text = _key(notes)
    return [f"{c.id}: evidence not found in the lesson notes" for c in vocabulary.concepts
            if _key(c.evidence) not in text]


def overlap_problems(vocabulary: Vocabulary) -> list[str]:
    """Names/aliases that point at more than one concept (the matcher could not tell them apart)."""
    owners: dict[str, set[str]] = {}
    for c in vocabulary.concepts:
        for term in (c.id, c.name, *c.aliases):
            owners.setdefault(_key(term), set()).add(c.id)
    return [f"'{term}' is used by {', '.join(sorted(ids))}" for term, ids in sorted(owners.items()) if len(ids) > 1]


def vocabulary_problems(vocabulary: Vocabulary, notes: str) -> list[str]:
    """Everything that blocks a draft or an approval: missing evidence and ambiguous names/aliases."""
    return evidence_problems(vocabulary, notes) + overlap_problems(vocabulary)


def _key(text: str) -> str:
    """Lowercase, punctuation-free, single-spaced form for comparisons."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower().replace("_", " ")).split())


class Match(BaseModel):
    concept_id: str | None
    method: Literal["exact", "alias", "embedding", "none"]
    similarity: float | None = None
    margin: float | None = None  # embedding only: best minus second-best similarity
    reason: str = ""  # why there is no match


class ConceptMatcher:
    """Maps free-text concept labels to a vocabulary's concept ids."""

    def __init__(self, vocabulary: Vocabulary, embedder: Any = None, min_similarity: float | None = None,
                 min_margin: float | None = None) -> None:
        self.vocabulary = vocabulary
        self.embedder = embedder
        self.min_similarity = settings.concept_match_min_similarity if min_similarity is None else min_similarity
        self.min_margin = settings.concept_match_min_margin if min_margin is None else min_margin
        self._names = {_key(c.name): c.id for c in vocabulary.concepts} | {_key(c.id): c.id for c in vocabulary.concepts}
        self._aliases = {_key(a): c.id for c in vocabulary.concepts for a in c.aliases}
        self._vectors: np.ndarray | None = None

    def match(self, label: str, context: str = "") -> Match:
        """Best concept for `label` (with optional `context`, e.g. the question text, for the embedding step)."""
        key = _key(label or "")
        if key in self._names:
            return Match(concept_id=self._names[key], method="exact", similarity=1.0)
        if key in self._aliases:
            return Match(concept_id=self._aliases[key], method="alias", similarity=1.0)
        if self.embedder is None or not key:
            return Match(concept_id=None, method="none",
                         reason="no embedder for a non-exact label" if key else "empty label")
        query = f"{label}. {context}".strip() if context else label
        sims = self._concept_vectors() @ np.asarray(self.embedder.embed_query(query), dtype="float32")
        order = np.argsort(sims)[::-1]
        best, similarity = int(order[0]), round(float(sims[order[0]]), 4)
        margin = round(float(sims[order[0]] - sims[order[1]]), 4) if len(order) > 1 else 1.0
        if similarity < self.min_similarity:
            return Match(concept_id=None, method="none", similarity=similarity, margin=margin,
                         reason=f"similarity {similarity:.3f} < {self.min_similarity}")
        if margin < self.min_margin:
            return Match(concept_id=None, method="none", similarity=similarity, margin=margin,
                         reason=f"ambiguous: {self.vocabulary.concepts[best].id} beats "
                                f"{self.vocabulary.concepts[int(order[1])].id} by only {margin:.3f}")
        return Match(concept_id=self.vocabulary.concepts[best].id, method="embedding", similarity=similarity,
                     margin=margin)

    def _concept_vectors(self) -> np.ndarray:
        if self._vectors is None:
            texts = [f"{c.name}: {c.description}" + (f" Also called: {', '.join(c.aliases)}." if c.aliases else "")
                     for c in self.vocabulary.concepts]
            self._vectors = np.asarray(self.embedder.embed_documents(texts), dtype="float32")
        return self._vectors
