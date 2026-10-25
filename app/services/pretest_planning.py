"""Choose a learner's pre-test questions from the topic bank and their earlier tests (pure functions, no I/O).

New learner on the topic: the first N bank items (the standard starting quiz).
Returning learner:
  1. up to ``retest_max`` questions whose latest attempt was wrong (has the learner learned them now?),
  2. unseen questions, those on concepts the learner got wrong first,
  3. only if nothing unseen is left: other previously wrong questions, then previously correct ones,
     least recently seen first.
``unseen_needed`` tells the caller how many new questions would have avoided step 3, so it can top up
the bank before planning again.
"""
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.services.mcq import normalize_answer, question_key

REASON_NEW_LEARNER = "Starting quiz for this topic."
REASON_RETEST = "Last time you found this, or a similar question on the same idea, difficult. Let's check it again."
REASON_WEAK = "New question on {concept}, which you found difficult."
REASON_UNSEEN = "New question."
REASON_REVIEW_WRONG = "You got this wrong before."
REASON_REVIEW_RIGHT = "Review: you answered this correctly before."



@dataclass(frozen=True)
class ItemHistory:
    """The learner's record on one bank item."""

    attempts: int
    last_correct: bool
    last_seen_at: datetime | None


@dataclass(frozen=True)
class PlannedItem:
    bank_index: int
    reason: str


@dataclass
class PretestPlan:
    mode: str  # "new" | "returning"
    items: list[PlannedItem]
    weak_concepts: list[str] = field(default_factory=list)
    unseen_needed: int = 0
    previous: dict[str, Any] | None = None


def item_histories(bank_items: Sequence[dict[str, Any]],
                   assessments: Sequence[dict[str, Any]]) -> dict[int, ItemHistory]:
    """Latest outcome per bank item; assessments are newest first and matched to the bank by question text.

    A paired post-test question (``form == "parallel"``, app/services/posttest.py) is a different question on the
    same concept and difficulty: its outcome counts for the pre-test item it is paired with (``parallel_to`` = that
    item's content id), so a learner who missed it is re-tested on that objective. It is scored with its own key.
    """
    index_by_key = {question_key(item["question"]): i for i, item in enumerate(bank_items)}
    index_by_id = {item["id"]: i for i, item in enumerate(bank_items) if item.get("id")}
    histories: dict[int, ItemHistory] = {}
    for assessment in assessments:
        answers = assessment["answers"]
        for position, item in enumerate(assessment["items"]):
            if item.get("form") == "parallel":
                index = index_by_id.get(item.get("parallel_to") or "")
            else:
                index = index_by_key.get(question_key(item["question"]))
            if index is None:
                continue
            answer = answers[position] if position < len(answers) else None
            correct = normalize_answer(answer) == item["answer_index"]
            known = histories.get(index)
            histories[index] = (ItemHistory(1, correct, assessment.get("created_at")) if known is None
                                else ItemHistory(known.attempts + 1, known.last_correct, known.last_seen_at))
    return histories


def concept_key(concept: str) -> str:
    """LLM-written concept names vary in case and spacing ("Nucleus", "NUCLEUS "); compare them loosely."""
    return " ".join(concept.lower().split())


def weak_concepts(bank_items: Sequence[dict[str, Any]], histories: dict[int, ItemHistory]) -> list[str]:
    """Concepts with at least one question whose latest attempt was wrong, in bank order (first spelling kept)."""
    weak: dict[str, str] = {}
    for index in sorted(histories):
        concept = bank_items[index]["concept"]
        if not histories[index].last_correct:
            weak.setdefault(concept_key(concept), concept)
    return list(weak.values())


def previous_session(assessments: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Pre/post scores of the learner's most recent session on the topic."""
    if not assessments:
        return None
    session_id = assessments[0]["session_id"]
    same = [a for a in assessments if a["session_id"] == session_id]
    score = {a["type"]: a["score"] for a in reversed(same)}  # newest wins
    return {"session_id": session_id, "date": assessments[0].get("created_at"),
            "pre_score": score.get("pre"), "post_score": score.get("post")}


def _recency(history: ItemHistory) -> float:
    return history.last_seen_at.timestamp() if history.last_seen_at else 0.0


def plan_pretest(bank_items: Sequence[dict[str, Any]], assessments: Sequence[dict[str, Any]], *,
                 n: int, retest_max: int) -> PretestPlan:
    """Pick up to ``n`` bank items for this learner's pre-test, each with the reason it was chosen."""
    if not assessments:
        chosen = [PlannedItem(i, REASON_NEW_LEARNER) for i in range(min(n, len(bank_items)))]
        return PretestPlan("new", chosen, unseen_needed=max(0, n - len(bank_items)))

    histories = item_histories(bank_items, assessments)
    weak = weak_concepts(bank_items, histories)
    weak_keys = {concept_key(c) for c in weak}
    wrong = sorted((i for i, h in histories.items() if not h.last_correct),
                   key=lambda i: (-_recency(histories[i]), i))
    chosen = [PlannedItem(i, REASON_RETEST) for i in wrong[:retest_max]][:n]

    unseen = sorted((i for i in range(len(bank_items)) if i not in histories),
                    key=lambda i: (concept_key(bank_items[i]["concept"]) not in weak_keys, i))
    for index in unseen[: n - len(chosen)]:
        concept = bank_items[index]["concept"]
        weak_concept = concept_key(concept) in weak_keys
        chosen.append(PlannedItem(index, REASON_WEAK.format(concept=concept) if weak_concept else REASON_UNSEEN))
    unseen_needed = max(0, n - len(chosen))

    if len(chosen) < n:  # bank exhausted: review what the learner has already seen
        taken = {p.bank_index for p in chosen}
        review_wrong = [i for i in wrong if i not in taken]
        review_right = sorted((i for i, h in histories.items() if h.last_correct),
                              key=lambda i: (_recency(histories[i]), i))
        for index in review_wrong:
            chosen.append(PlannedItem(index, REASON_REVIEW_WRONG))
        for index in review_right:
            chosen.append(PlannedItem(index, REASON_REVIEW_RIGHT))
        chosen = chosen[:n]

    return PretestPlan("returning", chosen, weak, unseen_needed, previous_session(assessments))
