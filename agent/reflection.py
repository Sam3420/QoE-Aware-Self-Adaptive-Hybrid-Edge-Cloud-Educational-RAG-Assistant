"""End-of-session reflection: goals met, mastery before vs after per concept, formative checks and coach suggestions.

`build_reflection` is pure; the reflect node (agent/nodes/coaching.py) feeds it the mastery recorded at planning
time and after the post-test. A goal is met when its mastery ends at MASTERED or higher, or rose by MET_GAIN or more
during the session (the Laplace estimate moves slowly: one right answer adds about 0.15).
"""
from typing import Any

from agent.coach import MASTERED

MET_GAIN = 0.15
WEAK = 0.5  # mastery below this (with evidence) is carried to the learner's next session


def build_reflection(goals: list[dict[str, Any]], before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]],
                     names: dict[str, str], suggestion_log: list[dict[str, Any]], formative_checks: int,
                     coach_mode: str) -> dict[str, Any]:
    """The reflection dict: summary text, per-goal outcome, per-concept mastery change and the session rates."""
    concepts = []
    for concept_id, end in after.items():
        start = before.get(concept_id) or {"mastery": 0.5, "attempts": 0}
        if end.get("attempts", 0) > start.get("attempts", 0) or any(g["concept_id"] == concept_id for g in goals):
            concepts.append({"concept_id": concept_id, "name": names.get(concept_id, concept_id),
                             "before": round(start["mastery"], 4), "after": round(end["mastery"], 4),
                             "gain": round(end["mastery"] - start["mastery"], 4),
                             "attempts": end.get("attempts", 0) - start.get("attempts", 0)})
    by_id = {c["concept_id"]: c for c in concepts}
    goal_rows = []
    for goal in goals:
        c = by_id.get(goal["concept_id"]) or {"before": goal["mastery"], "after": goal["mastery"], "gain": 0.0}
        met = c["after"] >= MASTERED or c["gain"] >= MET_GAIN
        goal_rows.append({"concept_id": goal["concept_id"], "name": goal["name"], "before": c["before"],
                          "after": c["after"], "gain": c["gain"], "met": met})
    shown = len(suggestion_log)
    accepted = sum(1 for e in suggestion_log if e["accepted"])
    practised = [c["gain"] for c in concepts if c["attempts"] > 0]
    weak_next = [c["concept_id"] for c in concepts if c["after"] < WEAK and (c["attempts"] or c["gain"] <= 0)]
    weak_next += [g["concept_id"] for g in goal_rows if not g["met"] and g["concept_id"] not in weak_next]
    reflection = {
        "coach_mode": coach_mode,
        "goals": goal_rows,
        "concepts": sorted(concepts, key=lambda c: -c["gain"]),
        "formative_checks": formative_checks,
        "suggestions_shown": shown,
        "suggestions_accepted": accepted,
        "suggestion_acceptance_rate": round(accepted / shown, 4) if shown else None,
        "goal_completion_rate": round(sum(g["met"] for g in goal_rows) / len(goal_rows), 4) if goal_rows else None,
        "mean_mastery_gain": round(sum(practised) / len(practised), 4) if practised else None,
        "weak_next_time": weak_next,
    }
    # TODO(teammate): call the shared reflection template here once it exists; this template is the fallback.
    reflection["summary"] = summary_text(reflection, names)
    return reflection


def summary_text(r: dict[str, Any], names: dict[str, str]) -> str:
    """A short templated summary for the learner (no LLM)."""
    lines = []
    if r["goals"]:
        met = [g["name"] for g in r["goals"] if g["met"]]
        lines.append(f"You met {len(met)} of your {len(r['goals'])} goals" + (f": {', '.join(met)}." if met else "."))
    gained = [c for c in r["concepts"] if c["gain"] > 0]
    if gained:
        best = gained[0]
        lines.append(f"Biggest step forward: {best['name']} ({best['before']:.0%} → {best['after']:.0%}).")
    if r["formative_checks"]:
        lines.append(f"You did {r['formative_checks']} practice check{'s' * (r['formative_checks'] != 1)}.")
    if r["weak_next_time"]:
        lines.append("Next time, let's work on " + ", ".join(names.get(c, c) for c in r["weak_next_time"][:3]) + ".")
    return " ".join(lines) or "Session complete."
