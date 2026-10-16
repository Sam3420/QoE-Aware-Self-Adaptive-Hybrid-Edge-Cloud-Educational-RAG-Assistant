"""Policy P5 (route "escalate"): local answers, escalated to the cloud on low retrieval confidence or a failed check."""
import json

import pytest

from agent.escalation import CheckResult, check_answer, reason_to_escalate_before
from services.llm_clients import GenerationResult, LLMError
from services.streaming import streaming_to
from services.vector_store import RetrievedChunk
from tests.conftest import pass_pretest, seed_experiences

QUESTION = "How do plants convert sunlight into chemical energy?"  # specific: strong fake retrieval
CTX = {"rtt_ms": 180.0, "cpu": 0.35, "mem": 0.55, "competency_num": 1.0, "last_qoe": 0.5}


class Verifier:
    """Fake check model replying with a fixed JSON verdict (or raising)."""

    def __init__(self, score=0.9, issue="", fail=False):
        self.reply = json.dumps({"score": score, "issue": issue})
        self.fail, self.requests = fail, []

    def generate(self, request):
        self.requests.append(request)
        if self.fail:
            raise LLMError("check model down")
        return GenerationResult(text=self.reply, provider="cloud", model="check-fake", latency_ms=5.0)


class FixedScoreBackend:
    """Retrieval that always scores `score`, even after corrective rewrites."""

    def __init__(self, score):
        self.score = score

    def search(self, resource_id, query, top_k):
        return [RetrievedChunk(chunk_id=f"w{i}", text="weak", score=self.score) for i in range(top_k)]

    def score_chunks(self, resource_id, query, chunk_ids):
        return {cid: self.score for cid in chunk_ids}


def _p5_session(runner, learner, services, verifier=None):
    """A session in which kNN picks P5, stopped at the first question."""
    seed_experiences(services, [("P5", 0.9), ("P5", 0.9), ("P1", 0.1)], context=CTX)
    services.verifier = verifier
    tid = runner.start_session(learner, "photosynthesis")["thread_id"]
    pass_pretest(runner, tid)
    values = runner.get_state(tid)["values"]
    assert values["policy_id"] == "P5" and values["runtime_config"]["route"] == "escalate"
    return tid


def _last_log(runner, tid):
    return runner.get_state(tid)["values"]["qa_log"][-1]


def test_confident_question_is_answered_locally_when_the_check_passes(runner, learner, services):
    verifier = Verifier(score=0.9)
    tid = _p5_session(runner, learner, services, verifier)
    runner.resume(tid, {"query": QUESTION})
    log = _last_log(runner, tid)
    assert log["model_used"].startswith("local:") and log["preferred_route"] == "escalate"
    assert (log["escalated"], log["check_supported"], log["check_score"]) == (False, True, 0.9)
    assert verifier.requests[0].operation == "verify-answer" and verifier.requests[0].temperature == 0.0
    assert QUESTION in verifier.requests[0].prompt and "Stub answer" in verifier.requests[0].prompt


def test_unsupported_local_answer_is_replaced_by_the_cloud_and_the_stream_is_reset(runner, learner, services):
    tid = _p5_session(runner, learner, services, Verifier(score=0.25, issue="claims glucose is a protein"))
    streamed = []
    with streaming_to(streamed.append):
        runner.resume(tid, {"query": QUESTION})
    log = _last_log(runner, tid)
    assert log["model_used"].startswith("cloud:") and log["escalated"] and log["escalation_stage"] == "after"
    assert log["draft_model"].startswith("local:") and log["check_supported"] is False
    assert "check score 0.25 < 0.75: claims glucose is a protein" == log["escalation_reason"]
    assert streamed[-2:] == [None, runner.get_state(tid)["values"]["response"]]  # draft cleared, cloud text shown
    assert "cloud-llm-fake" in streamed[-1]


def test_borderline_retrieval_goes_to_the_cloud_before_answering(runner, learner, services):
    verifier = Verifier(score=0.1)
    tid = _p5_session(runner, learner, services, verifier)
    services.crag.store = FixedScoreBackend(0.60)  # between the 0.55 floor and the 0.65 threshold
    runner.resume(tid, {"query": QUESTION})
    log = _last_log(runner, tid)
    assert log["model_used"].startswith("cloud:") and log["escalation_stage"] == "before"
    assert log["escalation_reason"].startswith("borderline retrieval confidence 0.60")
    assert verifier.requests == []  # cloud answers are not checked


def test_off_topic_question_stays_local_without_a_cloud_call(runner, learner, services):
    verifier = Verifier(score=0.9)
    tid = _p5_session(runner, learner, services, verifier)
    services.crag.store = FixedScoreBackend(0.40)  # below the floor: the video does not cover it
    runner.resume(tid, {"query": QUESTION})
    log = _last_log(runner, tid)
    assert log["model_used"].startswith("local:") and not log["escalated"]
    assert len(verifier.requests) == 1  # the local "not covered" answer is still checked


def test_failed_check_or_missing_verifier_keeps_the_local_answer(runner, learner, services):
    tid = _p5_session(runner, learner, services, Verifier(fail=True))
    runner.resume(tid, {"query": QUESTION})
    log = _last_log(runner, tid)
    assert log["model_used"].startswith("local:") and not log["escalated"] and log["check_supported"] is None
    assert "escalation check: LLMError: check model down" in log["error"]

    services.verifier = None
    runner.resume(tid, {"query": QUESTION})
    log = _last_log(runner, tid)
    assert log["model_used"].startswith("local:") and log["check_ms"] is None


def test_other_policies_never_check_or_escalate(runner, learner, services):
    verifier = Verifier(score=0.0)
    services.verifier = verifier
    tid = runner.start_session(learner, "photosynthesis")["thread_id"]
    pass_pretest(runner, tid)
    assert runner.get_state(tid)["values"]["policy_id"] == "P1"  # cold start default
    runner.resume(tid, {"query": QUESTION})
    log = _last_log(runner, tid)
    assert log["model_used"].startswith("local:") and not log["escalated"] and verifier.requests == []


def test_escalation_helpers():
    assert reason_to_escalate_before(0.70, 0.65) is None  # confident
    assert reason_to_escalate_before(0.50, 0.65) is None  # off-topic: below the 0.55 floor
    assert reason_to_escalate_before(0.60, 0.65) ==         "borderline retrieval confidence 0.60 (between 0.55 and threshold 0.65)"
    fenced = Verifier()
    fenced.reply = '```json\n{"score": 1.4, "issue": ""}\n```'  # fenced and out of range
    assert check_answer(fenced, "q", ["c"], "a").score == 1.0
    garbage = Verifier()
    garbage.reply = "looks fine to me"
    result = check_answer(garbage, "q", ["c"], "a")
    assert result.supported is None and "no JSON object" in result.error
    assert CheckResult().supported is None


@pytest.mark.parametrize("enabled,key,expected", [(True, "gsk_x", "openai/gpt-oss-20b"), (False, "gsk_x", None),
                                                  (True, None, None)])
def test_verifier_is_built_only_with_groq_and_when_enabled(enabled, key, expected):
    from agent.deps import make_verifier
    from config import Settings

    cfg = Settings(_env_file=None, escalation_check_enabled=enabled, **({"GROQ_API_KEY": key} if key else {}))
    verifier = make_verifier(cfg)
    assert (verifier.model if verifier else None) == expected
