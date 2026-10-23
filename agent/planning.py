"""Session planning rules: 2-4 learning goals per session, the recommendation query built from them, and progress.

Goal rules, in priority order (each concept once, at most PLAN_MAX_GOALS):
1. concepts of the questions answered wrong in this session's pre-test (in question order);
2. concepts with low mastery from earlier sessions (mastery < PLAN_LOW_MASTERY, lowest first, then most attempts);
3. only to reach PLAN_MIN_GOALS: concepts never practised (vocabulary order), then the weakest of the rest.

A test item is mapped to a concept through the topic's question_concept_map (by question text, reviewed mappings only)
and otherwise through its own concept label (name/alias exact match, then embeddings when an embedder is given).
"""
from typing import Any

from pydantic import BaseModel

from app.services.mcq import normalize_answer, question_key
from knowledge_prep.catalog import search_text
from knowledge_prep.concepts import ConceptMatcher, Vocabulary

MISSED = "missed in this pre-test"


class Goal(BaseModel):
    concept_id: str
    name: str
    description: str
    reason: str  # why it is a goal (shown on the plan card)
    mastery: float  # at planning time
    attempts: int


def item_concept(vocabulary: Vocabulary, item: dict[str, Any], embedder: Any = None) -> str | None:
    """The concept a test item assesses: reviewed question map first, then the item's concept label."""
    key = question_key(str(item.get("question") or ""))
    for mapping in vocabulary.question_concept_map.values():
        if question_key(mapping.question) == key and not mapping.needs_review:
            return mapping.concept_id
    return ConceptMatcher(vocabulary, embedder).match(str(item.get("concept") or ""),
                                                      context=str(item.get("question") or "")).concept_id


def missed_concepts(vocabulary: Vocabulary, assessment: dict[str, Any] | None, embedder: Any = None) -> list[str]:
    """Concepts of the wrongly answered items of a scored test, in question order (unmapped items skipped)."""
    if not assessment:
        return []
    answers = assessment.get("answers") or []
    out: list[str] = []
    for index, item in enumerate(assessment.get("items") or []):
        answer = answers[index] if index < len(answers) else None
        if normalize_answer(answer) == item.get("answer_index"):
            continue
        concept_id = item_concept(vocabulary, item, embedder)
        if concept_id is not None and concept_id not in out:
            out.append(concept_id)
    return out


def plan_goals(vocabulary: Vocabulary, missed: list[str], mastery: dict[str, Any], *, min_goals: int = 2,
               max_goals: int = 4, low_mastery: float = 0.5, carry_over: list[str] = ()) -> list[Goal]:
    """Apply the goal rules (module docstring). `mastery`: concept id -> ConceptState (tracker.get_mastery).
    `carry_over`: concepts the last session's reflection left weak (learner profile), right after the misses."""
    chosen: dict[str, str] = {}

    def add(concept_id: str, reason: str) -> None:
        if concept_id not in chosen and len(chosen) < max_goals and vocabulary.get(concept_id) is not None:
            chosen[concept_id] = reason

    for concept_id in missed:
        add(concept_id, MISSED)
    for concept_id in carry_over:
        add(concept_id, "still weak after your last session")
    weak = sorted((cid for cid, s in mastery.items() if s.attempts and s.mastery < low_mastery),
                  key=lambda cid: (mastery[cid].mastery, -mastery[cid].attempts))
    for concept_id in weak:
        s = mastery[concept_id]
        add(concept_id, f"low mastery {s.mastery:.2f} after {s.attempts} attempt{'s' * (s.attempts != 1)}")
    if len(chosen) < min_goals:
        for concept in vocabulary.concepts:
            if len(chosen) >= min_goals:
                break
            if not mastery[concept.id].attempts:
                add(concept.id, "not practised yet")
    if len(chosen) < min_goals:
        for concept_id in sorted(mastery, key=lambda cid: mastery[cid].mastery):
            if len(chosen) >= min_goals:
                break
            add(concept_id, f"weakest remaining (mastery {mastery[concept_id].mastery:.2f})")
    return [Goal(concept_id=cid, name=vocabulary.get(cid).name, description=vocabulary.get(cid).description,
                 reason=reason, mastery=round(mastery[cid].mastery, 4), attempts=mastery[cid].attempts)
            for cid, reason in chosen.items()]


def recommendation_query(topic_id: str, goals: list[Goal]) -> dict[str, Any]:
    """Arguments for VideoRecommender.recommend: the goals' names as concepts, and the objective they make
    (the same "<topic search text>: <concepts>" form the recommender builds itself)."""
    concepts = [g.name for g in goals]
    return {"concepts": concepts,
            "objective": f"{search_text(topic_id)}: {', '.join(concepts)}" if concepts else search_text(topic_id)}


def goal_progress(goals: list[dict[str, Any]], mastery: dict[str, Any]) -> dict[str, Any]:
    """Per-goal standing now (mastery change since planning, questions asked, video coverage) and a summary."""
    rows = []
    for goal in goals:
        s = mastery.get(goal["concept_id"])
        if s is None:
            continue
        status = ("practised" if s.attempts > goal["attempts"] else "discussed" if s.asked else
                  "covered by video" if s.covered else "not started")
        rows.append({"concept_id": goal["concept_id"], "name": goal["name"], "mastery_at_plan": goal["mastery"],
                     "mastery": round(s.mastery, 4), "asked": s.asked, "covered": s.covered, "status": status})
    touched = sum(r["status"] != "not started" for r in rows)
    untouched = [r["name"] for r in rows if r["status"] == "not started"]
    return {"goals": rows, "touched": touched, "total": len(rows),
            "suggestion": f"Ask about {untouched[0]} next." if untouched else "All goals have been worked on."}
