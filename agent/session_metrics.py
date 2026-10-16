"""Map what one learning session actually measured onto the QoE/SRE inputs (the s_next of a transition).

| Metric | Source in this project | When unavailable |
|---|---|---|
| LG | Hake gain (post − pre)/(1 − pre) from the pre/post-tests | no pre or post score |
| PT | post-test score | no post-test |
| RA | video-recommendation acceptance (accepted/shown) since the learner's previous session, same topic | no recommendations shown |
| LS | mean star rating of recommended videos in that window | no ratings |
| RQ | mean CRAG retrieval score per question (normalised with QOE_RETRIEVAL_FLOOR/CEIL) | no questions |
| LQ | Langfuse LLM-as-a-judge scores on the session's `generate-answer` generations (telemetry/answer_quality.py) | tracing off, LQ_SOURCE=off, no scores yet |
| SQ, TQ | none: speech-to-text / text-to-speech are not implemented | always |
| PL | mean end-to-end answer latency (cost: lower is better) | no questions |
| NR | successful network operations / attempted (answers without route/generation failure + RTT probes) | — |
| RU | peak of CPU and memory utilisation at start and end, as a health band | no samples |
| AV | answers from an LLM route (not the extractive fallback) / questions | no questions |
| SR | session star rating (1–5) | no rating |
| FS | lexicon sentiment of the session comment (estimate) | no comment / no lexicon words / disabled |
| CR | completed steps of the learning flow: pre-test (or already taken), ≥1 question, post-test | — |
| SRE CPU/MEM | mean of start and end utilisation (system-wide, psutil) | no samples |
| SRE EN | linear CPU power model × time the pipeline was busy (sum of answer latencies) — estimate | disabled / no questions |
| SRE NE | system-wide network bytes between context collection and evaluation (psutil) | counters unavailable |
"""
from datetime import datetime, timezone
from typing import Any

from app.db.repositories import RecommendationRepository
from config import Settings
from metrics.qoe import QoEBounds, QoERaw, normalize_qoe
from metrics.reward import RuntimeState
from metrics.sentiment import lexicon_sentiment
from metrics.sre import SREInputs, estimate_energy_wh
from services.monitoring import ProbeError
from services.youtube.metrics import recommendation_metrics


def qoe_bounds(cfg: Settings) -> QoEBounds:
    return QoEBounds(latency_good_ms=cfg.qoe_latency_good_ms, latency_bad_ms=cfg.qoe_latency_bad_ms,
                     retrieval_floor=cfg.qoe_retrieval_floor, retrieval_ceil=cfg.qoe_retrieval_ceil,
                     utilization_target=cfg.qoe_utilization_target,
                     utilization_saturation=cfg.qoe_utilization_saturation)


def sample_resources(monitor: Any, cfg: Settings) -> dict[str, Any]:
    """CPU, memory, network counter and one RTT probe, each None/False when unavailable."""
    sample: dict[str, Any] = {"cpu": None, "mem": None, "net_bytes": None, "probe_ok": None,
                              "at": datetime.now(timezone.utc).isoformat()}
    try:
        sample["cpu"], sample["mem"] = monitor.cpu_mem()
    except Exception:
        pass
    try:
        sample["net_bytes"] = monitor.net_bytes()
    except Exception:
        pass
    try:
        monitor.probe_rtt(cfg.probe_url, cfg.probe_timeout_s)
        sample["probe_ok"] = True
    except ProbeError:
        sample["probe_ok"] = False
    return sample


def recommendation_outcomes(session_factory: Any, student_id: str, topic_id: str,
                            since: datetime | None) -> tuple[float | None, float | None]:
    """(acceptance rate, mean star rating) of this learner's recommendation runs on the topic since `since`."""
    session = session_factory()
    try:
        repo = RecommendationRepository(session)
        items, events = [], []
        for run in repo.list_runs(student_id=student_id, limit=200):
            created = run["created_at"]
            if run["topic_id"] != topic_id or (since is not None and created is not None
                                               and created.replace(tzinfo=None) <= since.replace(tzinfo=None)):
                continue
            full = repo.get_run(run["id"])
            items += full["items"]
            events += repo.list_events(run["id"])
    finally:
        session.close()
    if not items:
        return None, None
    m = recommendation_metrics(items, events, None, None)
    satisfaction = m["learner_satisfaction"]["value"]
    return m["acceptance_rate"]["value"], None if satisfaction is None else satisfaction * 5


def build_next_state(state: dict[str, Any], *, end: dict[str, Any], session_factory: Any, cfg: Settings,
                     since: datetime | None, llm_quality: float | None = None) -> RuntimeState:
    """The measured post-adaptation state s_next of this session."""
    qa = state["qa_log"]
    start = state.get("resource_start") or {}
    context = state["context"]
    pre, post = state.get("pre_score"), state.get("post_score")
    gain = (post - pre) / (1 - pre) if pre is not None and post is not None and pre < 1 else None
    acceptance, item_stars = recommendation_outcomes(session_factory, state["student_id"], state["topic_id"], since)

    failures = sum(1 for e in qa if e.get("error") and ("unavailable" in e["error"] or "generation (" in e["error"]))
    probe_start_ok = context.get("rtt_ms") is not None and context.get("rtt_ms") != cfg.rtt_fail_ms
    probes = [ok for ok in (probe_start_ok, end.get("probe_ok")) if ok is not None]
    attempts = len(qa) + len(probes)
    reliability = (attempts - failures - probes.count(False)) / attempts if attempts else None

    loads = [max(c, m) for c, m in ((context.get("cpu"), context.get("mem")), (end.get("cpu"), end.get("mem")))
             if c is not None and m is not None]
    steps = [state.get("pre_score") is not None, bool(qa), post is not None]
    raw = QoERaw(
        learning_gain=gain,
        post_test=post,
        recommendation_acceptance=acceptance,
        learner_satisfaction_stars=item_stars,
        retrieval_score=sum(e["retrieval_score"] for e in qa) / len(qa) if qa else None,
        llm_quality=llm_quality,
        latency_ms=sum(e["latency_ms"] for e in qa) / len(qa) if qa else None,
        network_reliability=reliability,
        utilization=max(loads) if loads else None,
        availability=sum(1 for e in qa if not e["model_used"].startswith("extractive")) / len(qa) if qa else None,
        star_rating=state.get("rating"),
        sentiment=lexicon_sentiment(state.get("comment")) if cfg.sentiment_lexicon_enabled else None,
        completion=sum(steps) / len(steps),
    )

    cpus = [x for x in (context.get("cpu"), end.get("cpu")) if x is not None]
    mems = [x for x in (context.get("mem"), end.get("mem")) if x is not None]
    cpu_usage = sum(cpus) / len(cpus) if cpus else None
    busy_s = sum(e["latency_ms"] for e in qa) / 1000 if qa else None
    energy = (estimate_energy_wh(cpu_usage, busy_s, idle_power_w=cfg.energy_idle_power_w,
                                 max_power_w=cfg.energy_max_power_w) if cfg.energy_estimate_enabled else None)
    net_start, net_end = start.get("net_bytes"), end.get("net_bytes")
    bandwidth = net_end - net_start if net_start is not None and net_end is not None and net_end >= net_start else None
    sre = SREInputs(cpu_usage=cpu_usage, cpu_max=cfg.sre_cpu_max,
                    memory_usage=sum(mems) / len(mems) if mems else None, memory_max=cfg.sre_memory_max,
                    energy_wh=energy, energy_max_wh=cfg.sre_energy_max_wh, energy_estimated=energy is not None,
                    bandwidth_bytes=bandwidth, bandwidth_max_bytes=cfg.sre_bandwidth_max_bytes,
                    bandwidth_scope="system-wide (psutil)" if bandwidth is not None else None)

    next_context = {**context, "cpu": end.get("cpu", context.get("cpu")), "mem": end.get("mem", context.get("mem"))}
    return RuntimeState(context=next_context, qoe_metrics=normalize_qoe(raw, qoe_bounds(cfg)), sre_inputs=sre,
                        session_id=state["session_id"], measured_at=datetime.now(timezone.utc))
