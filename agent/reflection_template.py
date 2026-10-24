"""The shared end-of-session reflection template: the learner-facing summary of a reflection dict.

`render(reflection, names)` turns agent.reflection.build_reflection's output into a short Markdown paragraph:
how many goals were met (named), the biggest mastery changes per concept (before → after), any concept that
dropped, the practice done, and what to work on next. Deterministic, no LLM: the same session always gets the same
text, and it costs no latency. The reflection card (frontend) lists every goal and concept below this text, so the
paragraph only highlights; it does not repeat those lists.
"""
from typing import Any

MAX_HIGHLIGHTS = 2  # concepts named in the "biggest changes" sentence
MAX_NEXT = 3  # concepts named in the "next time" sentence


def render(reflection: dict[str, Any], names: dict[str, str]) -> str:
    """The summary paragraph for one finished session."""
    sentences = [_goals(reflection["goals"]), _changes(reflection["concepts"]), _practice(reflection),
                 _next(reflection, names)]
    return " ".join(s for s in sentences if s) or "Session complete. Ask a few questions next time so we can track your progress."


def _pct(value: float) -> str:
    return f"{value:.0%}"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" and {items[-1]}"


def _goals(goals: list[dict[str, Any]]) -> str:
    if not goals:
        return ""
    met = [f"**{g['name']}**" for g in goals if g["met"]]
    if len(met) == len(goals):
        return f"You met all {len(goals)} of your goals for this session: {_join(met)}."
    if met:
        return f"You met {len(met)} of your {len(goals)} goals ({_join(met)})."
    return f"You did not reach your {len(goals)} goals yet; that is normal for a first pass, and this session's answers show where to focus."


def _changes(concepts: list[dict[str, Any]]) -> str:
    rose = [c for c in concepts if c["gain"] > 0][:MAX_HIGHLIGHTS]
    fell = [c for c in concepts if c["gain"] < 0 and c.get("attempts", 0) > 0]
    parts = []
    if rose:
        parts.append("Your biggest step" + ("s" if len(rose) > 1 else "") + " forward: " + _join(
            [f"**{c['name']}** ({_pct(c['before'])} → {_pct(c['after'])})" for c in rose]) + ".")
    if fell:
        parts.append(_join([f"**{c['name']}**" for c in fell]) + " slipped"
                     + " a little, so it is worth another look.")
    return " ".join(parts)


def _practice(r: dict[str, Any]) -> str:
    bits = []
    if r.get("formative_checks"):
        n = r["formative_checks"]
        bits.append(f"{n} practice check{'s' if n != 1 else ''}")
    if r.get("suggestions_shown"):
        bits.append(f"{r['suggestions_accepted']} of {r['suggestions_shown']} suggestions taken")
    return f"Practice this session: {_join(bits)}." if bits else ""


def _next(r: dict[str, Any], names: dict[str, str]) -> str:
    weak = [names.get(c, c.replace("_", " ")) for c in r.get("weak_next_time") or []][:MAX_NEXT]
    if weak:
        return f"Next time, start with {_join([f'**{w}**' for w in weak])}: a short video and a quick check on each."
    if r.get("goals"):
        return "Next time, try a new topic or a harder video on this one."
    return ""
