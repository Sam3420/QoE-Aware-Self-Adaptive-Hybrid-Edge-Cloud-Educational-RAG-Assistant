"""Formative check items per concept (contract only; implemented on day 2, using app.services.mcq.extend_topic_items
to top up a concept that has too few questions)."""
from pydantic import BaseModel, Field


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

    Args:
        topic_id: catalogue topic id; its vocabulary must exist.
        concept_id: a concept of that vocabulary.
        n: how many items are wanted (fewer are returned when the pool cannot supply them).
        exclude_ids: Item.item_id values not to return.

    Raises:
        tracker.VocabularyNotFoundError: the topic has no vocabulary.
        tracker.UnknownConceptError: the concept is not in the vocabulary.
    """
    raise NotImplementedError
