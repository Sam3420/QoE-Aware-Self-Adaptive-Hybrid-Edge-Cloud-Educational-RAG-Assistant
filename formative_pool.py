"""Formative check items per concept: a pool of practice questions kept outside the topic question bank.

Why outside the bank: any bank item can be a pre- or post-test question, and the coaching node already excludes
every bank item from formative checks (agent/nodes/coaching.py). Questions appended to the bank with
extend_topic_items would therefore always be excluded, and a formative question that later became a pre/post-test
question would rehearse the test and inflate learning gain. The pool is generated with extend_topic_items' generator
(app.services.mcq.generate_concept_items: focused retrieval and prompt, avoid list) but stored in `formative_items`.

Each generated question is kept only if it is new (not in the bank or the pool, compared by normalised text) and
tests the requested concept: its own concept label names the concept, or matching.match_question maps its text to
the concept with at least COACH_MATCH_MIN_CONFIDENCE. Questions that clearly test another concept are dropped;
when no embedder is available the check is recorded as "unverified".
"""
import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

import tracker
from config import settings

logger = logging.getLogger(__name__)

__all__ = ["Item", "get_formative_items", "pool_size", "top_up"]


class Item(BaseModel):
    """One multiple-choice formative question on a concept (same fields as a question-bank item, plus ids)."""

    item_id: str  # stable id; pass it back in exclude_ids, and as the source_ref basis for tracker.record_attempt
    concept_id: str
    question: str
    options: list[str] = Field(min_length=2)
    answer_index: int = Field(ge=0)  # never send to the learner as-is
    explanation: str = ""


def get_formative_items(topic_id: str, concept_id: str, n: int, exclude_ids: list[str]) -> list[Item]:
    """Up to ``n`` questions testing ``concept_id``, skipping ``exclude_ids`` (e.g. items the learner has seen).

    Served from the stored pool; when it has fewer than ``n`` usable items the pool is topped up first (up to
    FORMATIVE_POOL_TARGET + n questions on the concept, one LLM call of MCQ_N_ITEMS questions at a time). If the
    top-up fails but some items exist, those are returned (fewer than ``n``); with none, the error is raised.

    Args:
        topic_id: catalogue topic id; its vocabulary must exist.
        concept_id: a concept of that vocabulary.
        n: how many items are wanted (fewer are returned when the pool cannot supply them).
        exclude_ids: Item.item_id values not to return.

    Raises:
        tracker.VocabularyNotFoundError: the topic has no vocabulary.
        tracker.UnknownConceptError: the concept is not in the vocabulary.
        app.services.errors.MCQGenerationError: the pool is empty and could not be generated.
    """
    tracker._check_concept(topic_id, concept_id)
    if n <= 0:
        return []
    excluded = set(exclude_ids)
    services = _services()
    with services.app._db() as db:
        pool = _pool(db, topic_id, concept_id)
        usable = [i for i in pool if i.item_id not in excluded]
        if len(usable) < n:
            # enough for this request, and at least FORMATIVE_POOL_TARGET on the concept for later ones
            want = max(n - len(usable), settings.formative_pool_target - len(pool))
            try:
                top_up(topic_id, concept_id, want, db=db)
            except Exception as exc:
                if not usable:
                    raise
                logger.warning("formative pool top-up failed for %s/%s; serving %d item(s): %s",
                               topic_id, concept_id, len(usable), exc)
            usable = [i for i in _pool(db, topic_id, concept_id) if i.item_id not in excluded]
    return usable[:n]


def pool_size(topic_id: str, concept_id: str, *, db: Session | None = None) -> int:
    """How many questions the pool holds for the concept."""
    if db is not None:
        return len(_pool(db, topic_id, concept_id))
    with _services().app._db() as session:
        return len(_pool(session, topic_id, concept_id))


def top_up(topic_id: str, concept_id: str, want: int, *, db: Session, llm: Any = None,
           max_calls: int = 3) -> dict[str, int]:
    """Generate questions on the concept until ``want`` new ones are stored (or ``max_calls`` LLM calls are used).

    Returns counts: {"added", "duplicates", "off_concept", "calls"}. Raises MCQGenerationError when the first call
    fails (later failures stop the top-up and keep what was added).
    """
    from app.db.models import FormativeItem
    from app.db.repositories import AdaptationRepository
    from app.services.mcq import generate_concept_items, question_key

    concept = next(c for c in tracker.get_concepts(topic_id) if c.id == concept_id)
    services = _services()
    llm = llm or services.app.mcq_llm
    bank = AdaptationRepository(db).get_question_bank(topic_id) or {}
    existing = [i["question"] for i in bank.get("items") or []] + [
        q for (q,) in db.execute(select(FormativeItem.question).where(FormativeItem.topic_id == topic_id))]
    seen = {question_key(q) for q in existing}
    position = len(_pool(db, topic_id, concept_id))
    counts = {"added": 0, "duplicates": 0, "off_concept": 0, "calls": 0}
    while counts["added"] < want and counts["calls"] < max_calls:
        counts["calls"] += 1
        try:
            items, result = generate_concept_items(topic_id, f"{concept.name}: {concept.description}",
                                                   avoid_questions=existing, retriever=services.app.retriever, llm=llm)
        except Exception:
            if counts["calls"] == 1:
                raise
            break
        for item in items:
            key = question_key(item["question"])
            if key in seen:
                counts["duplicates"] += 1
                continue
            check = _concept_check(item, topic_id, concept_id)
            if check == "other":
                counts["off_concept"] += 1
                continue
            seen.add(key)
            existing.append(item["question"])
            position += 1
            db.add(FormativeItem(id=f"fp_{uuid.uuid4().hex[:16]}", topic_id=topic_id, concept_id=concept_id,
                                 position=position, question=item["question"], question_key=key, options=list(item["options"]),
                                 answer_index=item["answer_index"], explanation=item.get("explanation") or "",
                                 concept_check=check, model_provider=result.model_provider,
                                 model_name=result.model_name))
            counts["added"] += 1
            if counts["added"] >= want:
                break
        db.flush()
    logger.info("formative pool %s/%s topped up: %s", topic_id, concept_id, counts)
    return counts


# --- internals -----------------------------------------------------------------------------------------------------

def _services() -> Any:
    from agent.deps import get_services

    return get_services()


def _pool(db: Session, topic_id: str, concept_id: str) -> list[Item]:
    from app.db.models import FormativeItem

    rows = db.scalars(select(FormativeItem).where(FormativeItem.topic_id == topic_id,
                                                  FormativeItem.concept_id == concept_id)
                      .order_by(FormativeItem.position))
    return [Item(item_id=r.id, concept_id=r.concept_id, question=r.question, options=list(r.options),
                 answer_index=r.answer_index, explanation=r.explanation) for r in rows]


def _concept_check(item: dict[str, Any], topic_id: str, concept_id: str) -> str:
    """label: the item's own concept label names the concept; matched: its text maps to the concept; other: it
    confidently maps to another concept; unverified: no confident match either way (e.g. no embedder)."""
    import matching
    from knowledge_prep.concepts import ConceptMatcher

    vocabulary = tracker._vocabulary(topic_id)
    label = ConceptMatcher(vocabulary).match(str(item.get("concept") or ""))  # names and aliases only
    if label.concept_id == concept_id:
        return "label"
    if label.concept_id is not None:  # the generator itself says it tests another concept
        return "other"
    found, confidence = matching.match_question(item["question"], topic_id)
    if confidence >= settings.coach_match_min_confidence:
        return "matched" if found == concept_id else "other"
    return "unverified"
