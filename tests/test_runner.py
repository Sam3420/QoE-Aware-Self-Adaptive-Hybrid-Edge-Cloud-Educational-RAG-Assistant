"""Runner API: start → resume through every interrupt → final state; errors for bad calls."""
import pytest

from agent.interfaces import LearnerNotFound
from agent.runner import SessionNotFound, SessionNotPaused
from tests.conftest import ANSWERS, pass_pretest


def test_full_session_through_runner(runner, learner, services):
    out = runner.start_session(learner, "photosynthesis")
    tid = out["thread_id"]
    assert out["interrupt"]["type"] == "pre_test"
    assert all("answer_index" not in item for item in out["interrupt"]["items"])  # answers never leak

    out = pass_pretest(runner, tid, ANSWERS)
    assert out["interrupt"] == {"type": "ask_question", "question_no": 1,
                                "prompt": 'Ask your question about this topic, or send {"end": true} to finish.'}
    out = runner.resume(tid, {"query": "How do leaves take in carbon dioxide?"})
    assert out["interrupt"]["question_no"] == 2
    out = runner.resume(tid, {"end": True})
    assert out["interrupt"]["type"] == "post_test_and_feedback" and len(out["interrupt"]["items"]) == 5

    state = runner.get_state(tid)
    assert state["next"] == ["post_test_and_feedback"] and not state["finished"]
    assert state["interrupt"]["type"] == "post_test_and_feedback"

    out = runner.resume(tid, {"answers": ANSWERS, "rating": 4, "comment": "clear"})
    final = out["final_state"]
    assert final["policy_id"] == "P1"
    assert final["decision_reason"].startswith("cold start")
    assert len(final["qa_log"]) == 1
    assert set(final["qa_log"][0]) >= {"query", "retrieval_score", "crag_retries", "model_used",
                                       "latency_ms", "error"}
    ev = final["evaluation"]
    assert ev["status"] == "completed" and ev["baseline_used"] is True  # first session: s is the baseline
    assert {"qoe_before", "qoe_after", "sre_before", "sre_after", "reward", "predicted_utility"} <= set(ev)
    assert ev["reward"] == round(0.7 * ev["delta_qoe"] + 0.3 * ev["delta_sre"], 6)
    assert runner.get_state(tid)["finished"]

    stored = services.app.get_experiences()
    assert [(e["session_id"], e["policy_id"]) for e in stored] == [(tid, "P1")]


def test_thread_id_is_the_learning_session_id(runner, learner, services):
    tid = runner.start_session(learner, "photosynthesis")["thread_id"]
    assert runner.get_state(tid)["values"]["session_id"] == tid


def test_unknown_learner_and_thread_and_finished_session(runner, learner):
    with pytest.raises(LearnerNotFound):
        runner.start_session("nobody", "photosynthesis")
    with pytest.raises(SessionNotFound):
        runner.get_state("missing")
    tid = runner.start_session(learner, "photosynthesis")["thread_id"]
    pass_pretest(runner, tid, ANSWERS)
    runner.resume(tid, {"end": True})
    runner.resume(tid, {"answers": ANSWERS, "rating": 3})
    with pytest.raises(SessionNotPaused):
        runner.resume(tid, {"query": "more?"})
