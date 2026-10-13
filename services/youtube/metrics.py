"""Recommendation evaluation metrics and QoE input signals (pure functions).

Every metric states its numerator and denominator and is `available=False` when the evidence is
missing. Opening a video is not watching it, and completion is self-reported, so it is labelled as such.
Learning gain is measured over the run's window and is not attributed to any single video.

QoE: per-run signals are normalised *inputs* grouped by the QoE model's components (EQ, AQ, NQ, UX);
they are not combined here. Session QoE is computed by metrics/qoe.py, whose RA (acceptance) and LS
(item ratings) inputs come from these runs (agent/session_metrics.py). The recommendation run's own
reward is still `reward_stub` below: recommendation runs have no before/after state yet.
"""
from collections.abc import Iterable, Mapping
from typing import Any

EVENT_TYPES = ("generated", "displayed", "opened", "accepted", "dismissed", "rated", "completed")
EXPLICIT_EVENTS = {"accepted", "dismissed", "rated"}


def _metric(value: float | None, numerator: float | None, denominator: float | None, definition: str,
            note: str | None = None) -> dict[str, Any]:
    return {"value": None if value is None else round(value, 4), "numerator": numerator,
            "denominator": denominator, "available": value is not None, "definition": definition, "note": note}


def _items_with(events: Iterable[Mapping[str, Any]], event_type: str) -> set[str]:
    return {e["item_id"] for e in events if e["event_type"] == event_type and e.get("item_id")}


def recommendation_metrics(items: list[Mapping[str, Any]], events: list[Mapping[str, Any]],
                           pre_score: float | None, post_score: float | None, k: int = 5) -> dict[str, dict[str, Any]]:
    """Compute run-level metrics from stored items, events and assessments around the run."""
    opened = _items_with(events, "opened")
    accepted = _items_with(events, "accepted")
    dismissed = _items_with(events, "dismissed") - accepted  # a later accept overrides a dismiss
    completed = _items_with(events, "completed")
    rated = _items_with(events, "rated")
    # An item the learner interacted with was necessarily shown, even if no "displayed" event arrived.
    displayed = _items_with(events, "displayed") | opened | accepted | dismissed | completed | rated
    latest_rating: dict[str, float] = {}
    for e in sorted(events, key=lambda e: str(e.get("created_at", ""))):
        if e["event_type"] == "rated" and e.get("item_id") and isinstance((e.get("value") or {}).get("rating"), (int, float)):
            latest_rating[e["item_id"]] = float(e["value"]["rating"])

    def ratio(num: set[str], den: set[str], definition: str, note: str | None = None) -> dict[str, Any]:
        n, d = len(num & den), len(den)
        return _metric(n / d if d else None, n, d, definition, note)

    top_k = {i["id"] for i in items if i.get("rank", k + 1) <= k}
    labelled = (accepted | dismissed) & top_k
    out = {
        "click_through_rate": ratio(opened, displayed, "items opened / items shown",
                                    "opening the embedded player or link is not evidence of watching"),
        "acceptance_rate": ratio(accepted, displayed, "items marked helpful / items shown"),
        "dismissal_rate": ratio(dismissed, displayed, "items dismissed / items shown"),
        "completion_rate": ratio(completed, opened, "items the learner reported finishing / items opened",
                                 "self-reported; watch time is not observable"),
        "learner_satisfaction": _metric(
            sum(latest_rating.values()) / len(latest_rating) / 5 if latest_rating else None,
            sum(latest_rating.values()) if latest_rating else None, 5 * len(latest_rating) if latest_rating else None,
            "mean item rating / 5 (latest rating per item)"),
        "precision_at_k": _metric(len(accepted & labelled) / len(labelled) if labelled else None,
                                  len(accepted & labelled), len(labelled),
                                  f"accepted / explicitly labelled items in the top {k} (unlabelled excluded)"),
    }
    gain = post_score - pre_score if pre_score is not None and post_score is not None else None
    out["learning_gain"] = _metric(gain, post_score, pre_score, "post-test − pre-test score for the topic",
                                   "measured around this run; not attributable to a single video")
    norm = gain / (1 - pre_score) if gain is not None and pre_score is not None and pre_score < 1 else None
    out["normalized_learning_gain"] = _metric(norm, gain, None if pre_score is None else 1 - pre_score,
                                              "(post − pre) / (1 − pre)  (Hake gain)")
    out["normalized_post_test"] = _metric(post_score, post_score, 1.0 if post_score is not None else None,
                                          "post-test score (0–1)")
    return out


def qoe_signals(metrics: Mapping[str, Mapping[str, Any]], pipeline: Mapping[str, Any], *,
                latency_budget_ms: float, rtt_ms: float | None, rtt_max_ms: float) -> dict[str, dict[str, float | None]]:
    """Normalised [0, 1] QoE inputs by component; None = not measurable for this run."""
    def value(name: str) -> float | None:
        return metrics.get(name, {}).get("value")

    timings = pipeline.get("timings") or {}
    total = timings.get("total_ms")
    rates = pipeline.get("rates") or {}
    failure = rates.get("api_failure_rate")
    norm_gain = value("normalized_learning_gain")
    return {
        "EQ": {
            "learning_gain": None if norm_gain is None else round(min(max((norm_gain + 1) / 2, 0.0), 1.0), 4),
            "normalized_post_test": value("normalized_post_test"),
        },
        "AQ": {
            "latency_quality": None if total is None else round(1 - min(total / latency_budget_ms, 1.0), 4),
            "api_reliability": None if failure is None else round(1 - failure, 4),
            "transcript_success_rate": rates.get("transcript_success_rate"),
        },
        "NQ": {"rtt_quality": None if rtt_ms is None else round(1 - min(rtt_ms / rtt_max_ms, 1.0), 4)},
        "UX": {
            "acceptance_rate": value("acceptance_rate"),
            "learner_satisfaction": value("learner_satisfaction"),
            "click_through_rate": value("click_through_rate"),
            "completion_rate": value("completion_rate"),
            "feedback_sentiment": None,  # no sentiment analyser exists in the project
        },
    }


def count_explicit_feedback(events: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Explicit judgements by kind: 'judged' (accepted/dismissed) and 'rated'. Displays and opens are not judgements."""
    return {"judged": sum(1 for e in events if e["event_type"] in {"accepted", "dismissed"}),
            "rated": sum(1 for e in events if e["event_type"] == "rated")}


def reward_stub(signals: Mapping[str, Mapping[str, float | None]], feedback: Mapping[str, int]) -> float | None:
    """STUB(Person B: metrics.reward): placeholder reward for a recommendation run.

    Uses acceptance only if the learner accepted/dismissed something, and satisfaction only if they
    rated: silence is not rejection, and a rating without a click is not a dismissal. None without
    any judgement, so such runs never influence strategy selection. Otherwise
    0.8 × mean of the available outcomes (acceptance, satisfaction, learning gain) + 0.2 × latency quality.
    """
    ux, eq, aq = signals.get("UX", {}), signals.get("EQ", {}), signals.get("AQ", {})
    explicit = []
    if feedback.get("judged", 0) and ux.get("acceptance_rate") is not None:
        explicit.append(ux["acceptance_rate"])
    if feedback.get("rated", 0) and ux.get("learner_satisfaction") is not None:
        explicit.append(ux["learner_satisfaction"])
    if not explicit:
        return None
    outcomes = explicit + ([eq["learning_gain"]] if eq.get("learning_gain") is not None else [])
    latency = aq.get("latency_quality")
    base = sum(outcomes) / len(outcomes)
    return round(0.8 * base + 0.2 * latency if latency is not None else base, 4)
