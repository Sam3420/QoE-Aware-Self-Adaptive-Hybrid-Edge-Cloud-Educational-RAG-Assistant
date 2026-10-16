"""LQ read back from Langfuse evaluator scores (fake Langfuse API client; nothing is sent over the network)."""
from types import SimpleNamespace as NS

import httpx
import pytest

from config import settings
from telemetry import tracing
from telemetry.answer_quality import session_answer_quality
from tests.conftest import ANSWERS, pass_pretest


def gen(trace_id, obs_id):
    return NS(trace_id=trace_id, id=obs_id)


def score(value, kind, subject_id):
    return NS(value=value, subject=NS(kind=kind, id=subject_id))


class FakeApi:
    """observations.get_many / scores.get_many over fixed data, recording the filters used."""

    def __init__(self, generations, scores, session_id):
        self.generations, self.scores_by_trace, self.calls = generations, scores, []
        self.session_id = session_id
        self.observations = NS(get_many=self._observations)
        self.scores = NS(get_many=self._scores)

    def _observations(self, **kw):
        self.calls.append(("observations", kw))
        return NS(data=[g for g in self.generations if kw["session_id"] == self.session_id], meta=NS(cursor=None))

    def _scores(self, **kw):
        self.calls.append(("scores", kw))
        return NS(data=[s for name, s in self.scores_by_trace.get(kw["trace_id"], []) if name == kw["name"]])


def client(generations, scores, session_id="s1"):
    return NS(api=FakeApi(generations, scores, session_id))


def test_mean_of_answers_from_generation_and_trace_scores():
    c = client([gen("t1", "g1"), gen("t2", "g2"), gen("t2", "g2-retry")],
               {"t1": [("answer_quality", score(0.9, "observation", "g1")),
                       ("answer_quality", score(0.1, "observation", "other-observation"))],  # not an answer
                "t2": [("answer_quality", score(0.5, "trace", "t2"))]})
    lq = session_answer_quality("s1", names=["answer_quality"], client=c)
    assert lq.value == pytest.approx((0.9 + 0.5) / 2) and (lq.scored, lq.answers) == (2, 2)
    obs_call = c.api.calls[0][1]
    assert obs_call["name"] == "generate-answer" and obs_call["session_id"] == "s1"
    assert all(kw["data_type"] == "NUMERIC" for kind, kw in c.api.calls if kind == "scores")


def test_judge_scale_is_normalised_and_score_names_averaged_per_answer():
    c = client([gen("t1", "g1")], {"t1": [("relevance", score(5, "observation", "g1")),
                                          ("groundedness", score(2, "observation", "g1"))]})
    lq = session_answer_quality("s1", names=["relevance", "groundedness"], score_min=1, score_max=5, client=c)
    assert lq.value == pytest.approx((1.0 + 0.25) / 2)


def test_partial_coverage_counts_only_scored_answers():
    c = client([gen("t1", "g1"), gen("t2", "g2")], {"t1": [("answer_quality", score(0.8, "observation", "g1"))]})
    lq = session_answer_quality("s1", names=["answer_quality"], client=c)
    assert lq.value == pytest.approx(0.8) and (lq.scored, lq.answers) == (1, 2)


@pytest.mark.parametrize("generations, scores, reason", [
    ([], {}, "no traced LLM answers"),
    ([gen("t1", "g1")], {}, "no 'answer_quality' evaluator scores yet"),
])
def test_unavailable_when_nothing_to_measure(generations, scores, reason):
    lq = session_answer_quality("s1", names=["answer_quality"], client=client(generations, scores))
    assert lq.value is None and reason in lq.reason


def test_unavailable_without_tracing_or_when_langfuse_is_down():
    assert session_answer_quality("s1", names=["answer_quality"]).reason == "Langfuse tracing is off"

    def down(**_):
        raise httpx.ConnectError("refused")

    broken = NS(api=NS(observations=NS(get_many=down), scores=NS(get_many=down)))
    lq = session_answer_quality("s1", names=["answer_quality"], client=broken)
    assert lq.value is None and "not reachable" in lq.reason


def test_waits_for_evaluators_until_every_answer_is_scored(monkeypatch):
    c = client([gen("t1", "g1")], {})
    sleeps = []

    def evaluator_finishes(seconds):
        sleeps.append(seconds)
        c.api.scores_by_trace["t1"] = [("answer_quality", score(0.7, "observation", "g1"))]

    monkeypatch.setattr("telemetry.answer_quality.time.sleep", evaluator_finishes)
    lq = session_answer_quality("s1", names=["answer_quality"], wait_s=5, client=c)
    assert lq.value == pytest.approx(0.7) and len(sleeps) == 1


def test_session_evaluation_uses_lq_from_langfuse(runner, learner, monkeypatch):
    seen = {}

    def fake_lq(session_id, **kw):
        seen["session_id"], seen["names"] = session_id, kw["names"]
        return session_answer_quality(session_id, client=client([gen("t1", "g1")], {"t1": [
            ("answer_quality", score(0.6, "observation", "g1"))]}, session_id), **kw)

    monkeypatch.setattr("agent.nodes.evaluation.session_answer_quality", fake_lq)
    monkeypatch.setattr("telemetry.answer_quality.session_answer_quality", fake_lq)
    tid = runner.start_session(learner, "photosynthesis")["thread_id"]
    pass_pretest(runner, tid)
    runner.resume(tid, {"query": "How do plants make glucose from light?"})
    runner.resume(tid, {"end": True})
    ev = runner.resume(tid, {"answers": ANSWERS, "rating": 4})["final_state"]["evaluation"]
    assert seen == {"session_id": tid, "names": settings.lq_score_names}
    assert ev["metrics"]["LQ"] == pytest.approx(0.6) and ev["lq"]["scored"] == 1


def test_lq_source_off_keeps_lq_unavailable(runner, learner, monkeypatch):
    monkeypatch.setattr(settings, "lq_source", "off")
    tracing.set_client(None)
    tid = runner.start_session(learner, "photosynthesis")["thread_id"]
    pass_pretest(runner, tid)
    runner.resume(tid, {"query": "How do plants make glucose from light?"})
    runner.resume(tid, {"end": True})
    ev = runner.resume(tid, {"answers": ANSWERS, "rating": 4})["final_state"]["evaluation"]
    assert ev["metrics"]["LQ"] is None and ev["lq"]["reason"] == "LQ_SOURCE=off"


def test_only_the_final_answer_of_a_question_is_scored():
    """P5 replaces an unsupported local answer with a cloud one: LQ scores what the learner ended up with."""
    from datetime import datetime, timedelta

    t0 = datetime(2026, 10, 17, 10, 0, 0)
    draft, final = NS(trace_id="t1", id="g-draft", start_time=t0), NS(trace_id="t1", id="g-final",
                                                                       start_time=t0 + timedelta(seconds=2))
    c = client([final, draft],  # API order does not matter, the later start time wins
               {"t1": [("answer_quality", score(0.2, "observation", "g-draft")),
                       ("answer_quality", score(0.9, "observation", "g-final"))]})
    lq = session_answer_quality("s1", names=["answer_quality"], client=c)
    assert lq.value == pytest.approx(0.9) and (lq.scored, lq.answers) == (1, 1)
