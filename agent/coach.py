"""Rule-based coach: after each answer, pick exactly one next-step suggestion (no LLM call).

Rules, first match wins (a suggestion the learner ignored twice in a row is muted and the search goes on):
1. formative_check: the last question was about a goal concept whose mastery is below LOW_MASTERY.
2. another_video:   a goal concept the current video does not cover (only when the video's coverage is known).
3. review:          the last answer was escalated (P5), failed the grounding check, or retrieval was weak.
4. post_test:       every goal has mastery of at least MASTERED.
5. follow_up:       the next goal not yet asked about or covered, else the weakest goal not yet mastered.

`choose_action` is pure: it takes plain data and returns the suggestion dict
``{action, concept_id, label, reason, query}`` (``query``: the question sent when a review/follow_up is accepted).
"""
from typing import Any

LOW_MASTERY = 0.5
MASTERED = 0.75
ACTIONS = ("formative_check", "another_video", "review", "post_test", "follow_up")


def choose_action(goals: list[dict[str, Any]], snapshot: dict[str, dict[str, Any]], *,
                  last_concept: str | None = None, last_answer: dict[str, Any] | None = None,
                  coverage: list[str] | None = None, muted: set[tuple[str, str | None]] = frozenset(),
                  low_mastery: float = LOW_MASTERY, mastered: float = MASTERED) -> dict[str, Any] | None:
    """The one suggestion for the learner's next step, or None when there are no goals (unplanned session).

    goals: session_plan["goals"] (concept_id, name). snapshot: concept id -> {mastery, attempts, correct, covered,
    asked}. last_concept: concept the last question was matched to (None if unknown or already practised).
    last_answer: {escalated, check_supported, retrieval_score, threshold} of the last answer. coverage: concept ids the
    current video covers, None if unknown. muted: (action, concept_id) pairs not to suggest again.
    """
    if not goals:
        return None
    names = {g["concept_id"]: g["name"] for g in goals}

    def mastery(concept_id: str) -> float:
        return float((snapshot.get(concept_id) or {}).get("mastery", 0.5))

    for candidate in _candidates(goals, snapshot, names, mastery, last_concept, last_answer or {}, coverage,
                                 low_mastery, mastered):
        if (candidate["action"], candidate["concept_id"]) not in muted:
            return candidate
    return None


SCAFFOLD_MISSES = 2  # formative misses on a concept that switch its answers to step-by-step, hint first
DEFAULT_SCAFFOLD = {"style": "concise", "support": "direct"}


def choose_scaffold(tracker_snapshot: dict[str, dict[str, Any]], concept_id: str | None,
                    profile: dict[str, Any]) -> dict[str, str]:
    """How to answer the current question: {style: concise|step_by_step, support: direct|hint_first}.

    1. SCAFFOLD_MISSES or more formative misses on the question's concept -> step_by_step + hint_first.
    2. Otherwise the learner profile's explanation style (free text from the profile page): "step"/"detail" ->
       step_by_step; "hint"/"guide"/"socratic"/"think" -> hint_first.
    3. Default: concise + direct.
    """
    misses = int((tracker_snapshot.get(concept_id) or {}).get("formative_misses", 0)) if concept_id else 0
    if misses >= SCAFFOLD_MISSES:
        return {"style": "step_by_step", "support": "hint_first", "why": f"{misses} practice misses on this concept"}
    stated = str(profile.get("explanation_style") or "").lower()
    if not stated:
        return {**DEFAULT_SCAFFOLD, "why": "default"}
    style = "step_by_step" if any(w in stated for w in ("step", "detail", "slow")) else "concise"
    support = "hint_first" if any(w in stated for w in ("hint", "guide", "socratic", "think")) else "direct"
    return {"style": style, "support": support, "why": f"your profile: {stated[:60]}"}


def muted_suggestions(log: list[dict[str, Any]], times: int = 2) -> set[tuple[str, str | None]]:
    """(action, concept_id) of a suggestion shown and not accepted at least `times` times in a row (latest first)."""
    if len(log) < times:
        return set()
    last = log[-times:]
    first = (last[0]["action"], last[0]["concept_id"])
    if all((e["action"], e["concept_id"]) == first and not e["accepted"] for e in last):
        return {first}
    return set()


def weakly_answered(last_answer: dict[str, Any]) -> str | None:
    """Why the last answer deserves a review, or None."""
    if last_answer.get("escalated"):
        return "the last answer had to be escalated to the cloud model"
    if last_answer.get("check_supported") is False:
        return "the last answer was only partly supported by the video"
    score, threshold = last_answer.get("retrieval_score"), last_answer.get("threshold")
    if score is not None and threshold is not None and score < threshold:
        return "the video only weakly matched your last question"
    return None


def _candidates(goals, snapshot, names, mastery, last_concept, last_answer, coverage, low_mastery, mastered):
    """Every applicable suggestion in rule order (choose_action takes the first one not muted)."""
    # 1. a goal concept just asked about, still weak
    if last_concept in names and mastery(last_concept) < low_mastery:
        name = names[last_concept]
        yield _make("formative_check", last_concept, f"Quick check: 2 questions on {name}",
                    f"You just asked about {name} and its mastery is {mastery(last_concept):.0%}.")
    # 2. goal concepts the current video does not cover
    if coverage is not None:
        for goal in goals:
            if goal["concept_id"] not in coverage:
                yield _make("another_video", goal["concept_id"], f"Find a video on {goal['name']}",
                            f"This video doesn't cover {goal['name']}, one of your goals.")
    # 3. the last answer was escalated or weakly grounded
    why = weakly_answered(last_answer)
    if why:
        if last_concept in names:
            name = names[last_concept]
            yield _make("review", last_concept, f"Explain {name} again, more simply",
                        f"Worth a second look: {why}.", query=f"Can you explain {name} again in simpler words?")
        else:
            yield _make("review", None, "Explain that again, more simply", f"Worth a second look: {why}.",
                        query="Can you explain your last answer again in simpler words, with an example?")
    # 4. every goal mastered
    if all(mastery(g["concept_id"]) >= mastered for g in goals):
        yield _make("post_test", None, "Take the post-test",
                    f"You've reached {mastered:.0%} mastery on every goal of this session.")
    # 5. next goal not explored yet, else the weakest unmastered goal
    untouched = [g for g in goals if not (snapshot.get(g["concept_id"]) or {}).get("asked")
                 and not (snapshot.get(g["concept_id"]) or {}).get("covered")]
    for goal in untouched:
        yield _make("follow_up", goal["concept_id"], f"Ask: What is {goal['name']}?",
                    f"{goal['name']} is a goal you haven't explored yet.", query=f"What is {goal['name']}?")
    for goal in sorted((g for g in goals if mastery(g["concept_id"]) < mastered),
                       key=lambda g: mastery(g["concept_id"])):
        yield _make("follow_up", goal["concept_id"], f"Ask: How does {goal['name']} work?",
                    f"{goal['name']} is your weakest goal ({mastery(goal['concept_id']):.0%} mastery).",
                    query=f"How does {goal['name']} work? Give an example.")


def _make(action: str, concept_id: str | None, label: str, reason: str, query: str = "") -> dict[str, Any]:
    return {"action": action, "concept_id": concept_id, "label": label, "reason": reason, "query": query}
