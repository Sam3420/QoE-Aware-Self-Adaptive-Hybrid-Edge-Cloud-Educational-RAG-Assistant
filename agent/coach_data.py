"""The coaching layer's only access to the data contract (tracker.py, matching.py, formative_pool.py).

Each function forwards to the real one with the agreed signature (COACH_DATA_SOURCE=real, the default). While a
real function is still a stub (raises NotImplementedError) a fallback stands in, so the graph runs end to end;
COACH_DATA_SOURCE=stub uses the fallbacks without calling matching / formative_pool at all (the switch back if the
real versions misbehave). The tracker is always the real one:
- match_question       -> (None, 0.0): no match, nothing is recorded
- video_coverage       -> None: coverage unknown, so the coach does not suggest another video
- get_formative_items  -> 2 fixed multiple-choice items on the concept (ids "stub:<concept_id>:<n>")
`implemented()` reports which functions are real, for logs and the UI.
"""
from typing import Any

import formative_pool
import matching
import tracker
from config import settings
from formative_pool import Item

__all__ = ["Item", "bank_item_id", "get_concepts", "get_formative_items", "get_mastery", "implemented",
           "mark_covered", "match_question", "record_attempt", "video_coverage"]

_STUBBED: set[str] = set()


def bank_item_id(topic_id: str, bank_position: int) -> str:
    """Id of a question-bank item (a pre/post-test question) in get_formative_items' exclude_ids."""
    return f"{topic_id}:{bank_position}"


# --- tracker (implemented) -------------------------------------------------------------------------------------------

def get_concepts(topic_id: str) -> list[Any]:
    return tracker.get_concepts(topic_id)


def record_attempt(learner_id: str, topic_id: str, concept_id: str, correct: bool, source: str, **kw: Any) -> None:
    """kw: tracker's optional keyword-only extras (source_ref for idempotency, db)."""
    tracker.record_attempt(learner_id, topic_id, concept_id, correct, source, **kw)


def mark_covered(learner_id: str, topic_id: str, concept_id: str, kind: str, **kw: Any) -> None:
    tracker.mark_covered(learner_id, topic_id, concept_id, kind, **kw)


def get_mastery(learner_id: str, topic_id: str, **kw: Any) -> dict[str, Any]:
    return tracker.get_mastery(learner_id, topic_id, **kw)


# --- matching / formative pool (stubbed until implemented) -----------------------------------------------------------

def _real() -> bool:
    return settings.coach_data_source == "real"


def match_question(text: str, topic_id: str) -> tuple[str | None, float]:
    try:
        if not _real():
            raise NotImplementedError
        return matching.match_question(text, topic_id)
    except NotImplementedError:
        _STUBBED.add("match_question")
        return None, 0.0


def video_coverage(video_id: str, topic_id: str) -> list[str] | None:
    """Concept ids the video covers, or None while coverage is unknown (stub)."""
    try:
        if not _real():
            raise NotImplementedError
        return matching.video_coverage(video_id, topic_id)
    except NotImplementedError:
        _STUBBED.add("video_coverage")
        return None


def get_formative_items(topic_id: str, concept_id: str, n: int, exclude_ids: list[str]) -> list[Item]:
    try:
        if not _real():
            raise NotImplementedError
        return formative_pool.get_formative_items(topic_id, concept_id, n, exclude_ids)
    except NotImplementedError:
        _STUBBED.add("get_formative_items")
        return [item for item in _stub_items(concept_id) if item.item_id not in set(exclude_ids)][:n]


def implemented() -> dict[str, bool]:
    """Which contract functions answered for real (False once a stub fallback was used in this process)."""
    return {name: name not in _STUBBED for name in ("match_question", "video_coverage", "get_formative_items")}


def _stub_items(concept_id: str) -> list[Item]:
    name = concept_id.replace("_", " ")
    return [
        Item(item_id=f"stub:{concept_id}:1", concept_id=concept_id,
             question=f"Which statement about {name} is correct? (practice question 1)",
             options=[f"It is a key idea of {name}.", "It has nothing to do with this topic.",
                      "It only applies to machines.", "It was disproved."],
             answer_index=0, explanation=f"{name.capitalize()} is one of the ideas this lesson teaches."),
        Item(item_id=f"stub:{concept_id}:2", concept_id=concept_id,
             question=f"Why does {name} matter in this chapter? (practice question 2)",
             options=["It does not matter.", f"It explains part of how the topic works ({name}).",
                      "It is only a historical fact.", "It is a spelling rule."],
             answer_index=1, explanation=f"The lesson uses {name} to explain how the topic works."),
    ]
