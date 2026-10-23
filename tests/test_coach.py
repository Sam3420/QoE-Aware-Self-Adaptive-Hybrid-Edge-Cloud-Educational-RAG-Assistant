"""Coach rules (agent/coach.py) and the coach / formative-check path through the graph, on the contract stubs
(agent/coach_data.py falls back to 2 fixed formative items while formative_pool is unimplemented)."""
from datetime import datetime, timezone

import pytest

import knowledge_prep.concepts as concepts_mod
from agent import coach_data
from agent.coach import choose_action, muted_suggestions
from knowledge_prep.concepts import Vocabulary, save_vocabulary
from tests.conftest import ANSWERS, pass_pretest

GOALS = [{"concept_id": "a", "name": "Alpha"}, {"concept_id": "b", "name": "Beta"}]
GOOD_ANSWER = {"escalated": False, "check_supported": True, "retrieval_score": 0.9, "threshold": 0.65}


def snap(**mastery):
    """concept -> state; a value is mastery, or (mastery, asked) to mark a goal as already asked about."""
    out = {}
    for cid, value in mastery.items():
        m, asked = value if isinstance(value, tuple) else (value, 0)
        out[cid] = {"mastery": m, "attempts": 2, "correct": 1, "covered": False, "asked": asked}
    return out


# --- rules -----------------------------------------------------------------------------------------------------------

def test_rule1_formative_check_when_the_concept_just_asked_about_is_weak():
    s = choose_action(GOALS, snap(a=0.3, b=0.9), last_concept="a", last_answer=GOOD_ANSWER)
    assert (s["action"], s["concept_id"]) == ("formative_check", "a") and "30%" in s["reason"]
    assert choose_action(GOALS, snap(a=0.6, b=0.9), last_concept="a", last_answer=GOOD_ANSWER)["action"] != \
        "formative_check"  # not weak
    assert choose_action(GOALS, snap(a=0.3, b=0.9), last_concept="zzz", last_answer=GOOD_ANSWER)["action"] != \
        "formative_check"  # not a goal


def test_rule2_another_video_for_a_goal_the_video_does_not_cover():
    s = choose_action(GOALS, snap(a=0.6, b=0.6), coverage=["a"], last_answer=GOOD_ANSWER)
    assert (s["action"], s["concept_id"]) == ("another_video", "b")
    assert choose_action(GOALS, snap(a=0.6, b=0.6), coverage=None, last_answer=GOOD_ANSWER)["action"] != \
        "another_video"  # coverage unknown (stub): never suggested


@pytest.mark.parametrize("answer, why", [
    ({**GOOD_ANSWER, "escalated": True}, "escalated"),
    ({**GOOD_ANSWER, "check_supported": False}, "only partly supported"),
    ({**GOOD_ANSWER, "retrieval_score": 0.5}, "weakly matched"),
])
def test_rule3_review_after_an_escalated_or_weakly_grounded_answer(answer, why):
    s = choose_action(GOALS, snap(a=0.6, b=0.6), last_concept="a", last_answer=answer)
    assert (s["action"], s["concept_id"]) == ("review", "a") and why in s["reason"]
    assert s["query"] == "Can you explain Alpha again in simpler words?"
    unmatched = choose_action(GOALS, snap(a=0.6, b=0.6), last_concept=None, last_answer=answer)
    assert unmatched["action"] == "review" and unmatched["concept_id"] is None


def test_rule4_post_test_when_every_goal_is_mastered():
    s = choose_action(GOALS, snap(a=0.8, b=0.75), last_answer=GOOD_ANSWER)
    assert s["action"] == "post_test" and s["concept_id"] is None


def test_rule5_follow_up_on_the_next_unexplored_goal_then_the_weakest():
    s = choose_action(GOALS, snap(a=(0.6, 1), b=0.6), last_answer=GOOD_ANSWER)
    assert (s["action"], s["concept_id"], s["label"]) == ("follow_up", "b", "Ask: What is Beta?")
    assert s["query"] == "What is Beta?"
    s = choose_action(GOALS, snap(a=(0.7, 1), b=(0.6, 2)), last_answer=GOOD_ANSWER)  # both explored
    assert (s["action"], s["concept_id"]) == ("follow_up", "b") and "weakest" in s["reason"]


def test_rule_priority_first_match_wins():
    weak_escalated = {**GOOD_ANSWER, "escalated": True}
    every_rule = dict(last_concept="a", last_answer=weak_escalated, coverage=[])
    assert choose_action(GOALS, snap(a=0.3, b=0.3), **every_rule)["action"] == "formative_check"  # 1 over 2, 3, 5
    assert choose_action(GOALS, snap(a=0.6, b=0.3), **every_rule)["action"] == "another_video"  # 2 over 3, 5
    assert choose_action(GOALS, snap(a=0.6, b=0.3), last_concept="a", last_answer=weak_escalated,
                         coverage=["a", "b"])["action"] == "review"  # 3 over 5
    assert choose_action(GOALS, snap(a=0.9, b=0.9), last_concept="a", last_answer=weak_escalated,
                         coverage=["a", "b"])["action"] == "review"  # 3 over 4
    assert choose_action(GOALS, snap(a=0.9, b=0.9), last_answer=GOOD_ANSWER)["action"] == "post_test"  # 4 over 5


def test_a_suggestion_ignored_twice_in_a_row_is_muted():
    log = [{"action": "follow_up", "concept_id": "b", "accepted": False}] * 2
    assert muted_suggestions(log) == {("follow_up", "b")}
    assert muted_suggestions(log[:1]) == set()
    assert muted_suggestions([*log[:1], {"action": "follow_up", "concept_id": "b", "accepted": True}]) == set()
    s = choose_action(GOALS, snap(a=(0.7, 1), b=0.6), last_answer=GOOD_ANSWER, muted=muted_suggestions(log))
    # every (follow_up, b) candidate is skipped, so the coach moves on to the other goal
    assert (s["action"], s["concept_id"]) == ("follow_up", "a") and "weakest" in s["reason"]


def test_no_goals_no_suggestion():
    assert choose_action([], {}) is None


# --- graph, on stubs -------------------------------------------------------------------------------------------------

TOPIC = "photosynthesis"  # FakeMCQProvider labels its 5 bank items "concept 1".."concept 5"


@pytest.fixture
def vocab_dir(tmp_path, monkeypatch):
    concepts = [{"id": f"concept_{i}", "name": f"Concept {i}", "description": f"What concept {i} is about.",
                 "aliases": [], "evidence": "Plants make food from light."} for i in range(1, 7)]
    monkeypatch.setattr(concepts_mod, "CONCEPTS_DIR", tmp_path)
    save_vocabulary(Vocabulary(topic_id=TOPIC, generated_by="test",
                               generated_at=datetime(2026, 10, 23, tzinfo=timezone.utc), concepts=concepts))


@pytest.fixture
def matched(monkeypatch):
    """match_question stands in for the real matcher: every question is about concept_1."""
    monkeypatch.setattr("matching.match_question", lambda text, topic_id: ("concept_1", 0.9))


@pytest.fixture
def formative_calls(monkeypatch):
    calls, original = [], coach_data.get_formative_items

    def spy(topic_id, concept_id, n, exclude_ids):
        calls.append({"topic_id": topic_id, "concept_id": concept_id, "n": n, "exclude_ids": list(exclude_ids)})
        return original(topic_id, concept_id, n, exclude_ids)

    monkeypatch.setattr(coach_data, "get_formative_items", spy)
    return calls


def _first_answer(runner, learner):
    tid = runner.start_session(learner, TOPIC)["thread_id"]
    pass_pretest(runner, tid, ["B", *ANSWERS[1:]])  # concept 1 missed -> mastery 1/3, a session goal
    return tid, runner.resume(tid, {"query": "How do plants make glucose from light?"})


def test_formative_check_round_trip(runner, learner, services, vocab_dir, matched, formative_calls, visited):
    tid, out = _first_answer(runner, learner)
    suggestion = out["interrupt"]["suggestion"]
    assert suggestion["action"] == "formative_check" and suggestion["concept_id"] == "concept_1"
    assert set(suggestion) == {"action", "concept_id", "label", "reason"}

    visited.clear()
    out = runner.resume(tid, {"accept_suggestion": True})
    first = out["interrupt"]
    assert first["type"] == "formative_check" and (first["index"], first["total"]) == (0, 2)
    assert "answer_index" not in first["item"] and first["feedback"] is None  # the key never reaches the learner
    call = formative_calls[0]
    assert (call["concept_id"], call["n"]) == ("concept_1", 2)
    assert {f"{TOPIC}:{i}" for i in range(5)} <= set(call["exclude_ids"])  # every pre/post bank question

    out = runner.resume(tid, {"answer": "A"})  # stub item 1: A is right
    second = out["interrupt"]
    assert second["index"] == 1 and second["feedback"]["correct"] is True and second["feedback"]["explanation"]
    out = runner.resume(tid, {"answer": "C"})  # stub item 2: B is right
    back = out["interrupt"]
    assert back["type"] == "ask_question"
    # nodes are recorded when they finish: formative_check once (after its last answer), ask_question still pending
    assert visited == ["ask_question", "plan_formative", "formative_check", "record_formative", "update_tracker",
                       "monitor_progress", "coach"]
    result = back["formative_result"]
    assert (result["correct"], result["total"]) == (1, 2) and result["results"][1]["answer"] == "B"

    values = runner.get_state(tid)["values"]
    concept = values["tracker_snapshot"]["concept_1"]
    assert (concept["attempts"], concept["correct"]) == (3, 1)  # pre-test miss + 1 right + 1 wrong
    assert back["suggestion"]["action"] != "formative_check" or back["suggestion"]["concept_id"] != "concept_1"
    assert values["suggestion_log"] == [{"question_no": 1, "action": "formative_check", "concept_id": "concept_1",
                                         "accepted": True}]


def test_formative_items_are_not_repeated_in_a_session(runner, learner, services, vocab_dir, matched,
                                                       formative_calls):
    tid, _ = _first_answer(runner, learner)
    runner.resume(tid, {"accept_suggestion": True})
    runner.resume(tid, {"answer": "B"})
    runner.resume(tid, {"answer": "A"})  # both wrong: concept_1 stays weak
    out = runner.resume(tid, {"query": "Tell me more about how plants make glucose."})
    assert out["interrupt"]["suggestion"]["action"] == "formative_check"
    runner.resume(tid, {"accept_suggestion": True})
    second_call = formative_calls[1]
    assert {"stub:concept_1:1", "stub:concept_1:2"} <= set(second_call["exclude_ids"])
    values = runner.get_state(tid)["values"]
    assert values["formative"]["items"] == []  # the stub pool has nothing new left
    out = runner.get_state(tid)
    assert out["interrupt"]["type"] == "ask_question" and "no practice questions" in out["interrupt"][
        "formative_result"]["note"]


def test_ignored_suggestions_are_logged_and_summarised(runner, learner, vocab_dir):
    tid, out = _first_answer(runner, learner)  # match_question is a stub: rule 5 follow_up
    assert out["interrupt"]["suggestion"]["action"] == "follow_up"
    runner.resume(tid, {"query": "Why are leaves green?"})  # ignored
    runner.resume(tid, {"end": True})  # ignored again
    final = runner.resume(tid, {"answers": ANSWERS, "rating": 4})["final_state"]
    log = final["suggestion_log"]
    assert [e["accepted"] for e in log] == [False, False]
    assert final["evaluation"]["coach"]["shown"] == 2 and final["evaluation"]["coach"]["acceptance_rate"] == 0.0


def test_accepted_follow_up_asks_the_templated_question(runner, learner, vocab_dir):
    tid, out = _first_answer(runner, learner)
    suggestion = out["interrupt"]["suggestion"]
    runner.resume(tid, {"accept_suggestion": True})
    values = runner.get_state(tid)["values"]
    assert values["qa_log"][-1]["query"] == f"What is {suggestion['label'].removeprefix('Ask: What is ')[:-1]}?"
    assert values["suggestion_log"][-1]["accepted"] is True


def test_accepted_another_video_searches_for_that_concept(runner, learner, services, vocab_dir, monkeypatch):
    monkeypatch.setattr("matching.video_coverage", lambda video_id, topic_id: [])  # the video covers no goal
    calls, original = [], services.videos.recommend
    services.videos.recommend = lambda *a, **kw: (calls.append(kw), original(*a, **kw))[1]
    tid, out = _first_answer(runner, learner)
    suggestion = out["interrupt"]["suggestion"]
    assert suggestion["action"] == "another_video"
    out = runner.resume(tid, {"accept_suggestion": True})
    assert out["interrupt"]["type"] == "select_video"
    goal_name = next(g["name"] for g in runner.get_state(tid)["values"]["session_plan"]["goals"]
                     if g["concept_id"] == suggestion["concept_id"])
    assert calls[-1]["concepts"] == [goal_name] and calls[-1]["objective"].endswith(f": {goal_name}")
    assert runner.get_state(tid)["values"]["video_focus"] == ""  # used once


def test_accept_without_a_suggestion_is_re_asked(runner, learner):
    tid = runner.start_session(learner, TOPIC)["thread_id"]  # no vocabulary: unplanned, no suggestions
    out = pass_pretest(runner, tid)
    assert out["interrupt"]["suggestion"] is None
    out = runner.resume(tid, {"accept_suggestion": True})
    assert out["interrupt"]["type"] == "ask_question" and "no suggestion" in out["interrupt"]["error"]


# --- the Streamlit page, end to end on stubs -------------------------------------------------------------------------

def coach_page():
    """The real learning-session page over the real graph on fake services (kept across reruns)."""
    import threading

    import streamlit as st
    from langgraph.checkpoint.memory import InMemorySaver

    from agent.graph import build_graph
    from agent.runner import SessionRunner
    from frontend.clients.agent_backend import AgentBackend
    from frontend.utils import state as ui_state
    from frontend.views import learning_session
    from tests.conftest import ANSWERS, pass_pretest

    if "coach_backend" not in st.session_state:
        from agent.deps import get_services

        backend = AgentBackend.__new__(AgentBackend)  # skip __init__: no real DB, checkpoint file or warm-up
        backend.services, backend._lock = get_services(), threading.Lock()
        backend.runner = SessionRunner(build_graph(InMemorySaver()))
        learner = backend.services.app.create_learner("Page learner")
        tid = backend.runner.start_session(learner, "photosynthesis")["thread_id"]
        pass_pretest(backend.runner, tid, ["B", *ANSWERS[1:]])
        backend.runner.resume(tid, {"query": "How do plants make glucose from light?"})
        st.session_state["coach_backend"] = backend
        st.session_state["coach_learner"] = {"id": learner, "display_name": "Page learner"}
        ui_state.put(ui_state.AGENT_SESSION, tid)
    learning_session.render(st.session_state["coach_backend"], st.session_state["coach_learner"])


def test_suggestion_button_runs_a_formative_check_in_the_ui(services, vocab_dir, matched):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(coach_page, default_timeout=60).run()
    assert not at.exception
    suggestion = next(b for b in at.button if b.label.startswith("Quick check"))
    assert any("mastery is 33%" in c.value for c in at.caption)  # the reason line
    assert {"Watch another video", "Take the post-test"} <= {b.label for b in at.button}  # existing buttons stay

    at = suggestion.click().run()
    assert not at.exception and at.subheader[0].value == "Quick check: Concept 1"
    at.radio[0].set_value(0)
    at = at.button[0].click().run()  # "Check my answer": A is right for stub item 1
    assert not at.exception and any("Correct" in s.value for s in at.success)
    at.radio[0].set_value(2)
    at = at.button[0].click().run()  # C is wrong for stub item 2
    assert not at.exception
    assert any("Practice check on Concept 1: 1 / 2 correct" in m.value for m in at.markdown)
    assert any("Not quite" in e.value for e in at.error)
    assert not any(b.label.startswith("Quick check") for b in at.button)  # a new suggestion, not the same check
    backend = at.session_state["coach_backend"]
    log = backend.runner.get_state(at.session_state["session"])["values"]["suggestion_log"]
    assert log == [{"question_no": 1, "action": "formative_check", "concept_id": "concept_1", "accepted": True}]
