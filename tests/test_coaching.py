"""Session planning rules (agent/planning.py) and the coaching nodes in the graph: plan_session before the videos,
update_tracker and monitor_progress after each answer (fake services, fixture vocabulary; matching.py is a stub)."""
from datetime import datetime, timezone

import pytest

import knowledge_prep.concepts as concepts_mod
from agent.planning import MISSED, goal_progress, missed_concepts, plan_goals, recommendation_query
from knowledge_prep.concepts import QuestionMapping, Vocabulary, save_vocabulary
from tests.conftest import ANSWERS, pass_pretest
from tracker import ConceptState

TOPIC = "photosynthesis"  # the fake services' topic; FakeMCQProvider labels its items "concept 1".."concept 5"


def vocabulary(**kw) -> Vocabulary:
    concepts = [{"id": f"concept_{i}", "name": f"Concept {i}", "description": f"What concept {i} is about.",
                 "aliases": [], "evidence": "Plants make food from light."} for i in range(1, 7)]
    return Vocabulary(topic_id=TOPIC, generated_by="test", generated_at=datetime(2026, 10, 23, tzinfo=timezone.utc),
                      concepts=concepts, **kw)


def state(mastery=0.5, attempts=0, asked=0, covered=False):
    correct = round(mastery * (attempts + 2) - 1)
    return ConceptState(mastery=mastery, attempts=attempts, correct=max(correct, 0), covered=covered, asked=asked)


def fresh(**overrides):
    return {f"concept_{i}": overrides.get(f"concept_{i}", state()) for i in range(1, 7)}


# --- rules -----------------------------------------------------------------------------------------------------------

def test_pretest_misses_come_first_then_low_mastery_lowest_first():
    mastery = fresh(concept_3=state(0.25, 2), concept_4=state(0.4, 3), concept_6=state(0.75, 2))
    goals = plan_goals(vocabulary(), ["concept_2"], mastery)
    assert [g.concept_id for g in goals] == ["concept_2", "concept_3", "concept_4"]
    assert goals[0].reason == MISSED and goals[1].reason == "low mastery 0.25 after 2 attempts"


def test_at_most_four_goals():
    mastery = fresh(**{f"concept_{i}": state(0.25, 2) for i in range(1, 7)})
    assert len(plan_goals(vocabulary(), ["concept_6", "concept_5"], mastery)) == 4


def test_at_least_two_goals_padded_with_unpractised_then_weakest():
    goals = plan_goals(vocabulary(), [], fresh(concept_1=state(0.75, 2)))
    assert [(g.concept_id, g.reason) for g in goals] == [("concept_2", "not practised yet"),
                                                         ("concept_3", "not practised yet")]
    everything_practised = {f"concept_{i}": state(0.5 + i / 20, 2) for i in range(1, 7)}
    goals = plan_goals(vocabulary(), [], everything_practised)  # nothing weak (all >= 0.5), nothing new
    assert [g.concept_id for g in goals] == ["concept_1", "concept_2"] and "weakest remaining" in goals[0].reason


def test_missed_concepts_use_reviewed_map_then_labels():
    items = [{"question": "Q one?", "answer_index": 0, "concept": "concept 3"},
             {"question": "Q two?", "answer_index": 1, "concept": "Concept 4"},
             {"question": "Q three?", "answer_index": 2, "concept": "something vague"},
             {"question": "Q four?", "answer_index": 0, "concept": "concept 6"}]
    vocab = vocabulary(question_concept_map={
        "0": QuestionMapping(concept_id="concept_1", question="Q one?", confidence=0.9),  # map beats the label
        "3": QuestionMapping(concept_id="concept_2", question="Q four?", confidence=0.3, needs_review=True)})
    assessment = {"items": items, "answers": ["B", "A", "A", "B"]}  # all four wrong
    # q1 via map, q2 via label, q3 unmapped (skipped), q4: unreviewed mapping ignored -> its label
    assert missed_concepts(vocab, assessment) == ["concept_1", "concept_4", "concept_6"]
    assert missed_concepts(vocab, {"items": items, "answers": ["A", "B", "C", "A"]}) == []  # all right
    assert missed_concepts(vocab, None) == []


def test_recommendation_query_and_progress():
    goals = plan_goals(vocabulary(), ["concept_2", "concept_5"], fresh())
    query = recommendation_query("life_processes", goals)
    assert query == {"concepts": ["Concept 2", "Concept 5"],
                     "objective": "Life Processes class 10: Concept 2, Concept 5"}
    planned = [g.model_dump() for g in goals]
    progress = goal_progress(planned, fresh(concept_2=state(0.5, 0, asked=1)))
    assert [g["status"] for g in progress["goals"]] == ["discussed", "not started"]
    assert progress["touched"] == 1 and progress["suggestion"] == "Ask about Concept 5 next."


# --- nodes in the graph ----------------------------------------------------------------------------------------------

@pytest.fixture
def vocab_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(concepts_mod, "CONCEPTS_DIR", tmp_path)
    save_vocabulary(vocabulary())
    return tmp_path


@pytest.fixture
def recommend_calls(services):
    calls, original = [], services.videos.recommend

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    services.videos.recommend = spy
    return calls


def test_plan_from_pretest_misses_shapes_the_video_query(runner, learner, services, vocab_dir, recommend_calls,
                                                         visited):
    tid = runner.start_session(learner, TOPIC)["thread_id"]
    wrong_1_and_5 = ["B", *ANSWERS[1:4], "B"]  # FakeMCQProvider: items 1 and 5 test "concept 1" and "concept 5"
    pass_pretest(runner, tid, wrong_1_and_5)
    values = runner.get_state(tid)["values"]
    plan = values["session_plan"]
    assert plan["status"] == "ok" and plan["used_pretest"] and plan["pretest_missed"] == ["concept_1", "concept_5"]
    assert [(g["concept_id"], g["reason"]) for g in plan["goals"]] == [("concept_1", MISSED), ("concept_5", MISSED)]
    assert plan["goals"][0]["mastery"] == pytest.approx(1 / 3, abs=1e-4)  # the live pre-test recording already counted it
    assert recommend_calls[0]["concepts"] == ["Concept 1", "Concept 5"]
    assert recommend_calls[0]["objective"].endswith(": Concept 1, Concept 5")
    assert visited.index("pre_test") < visited.index("plan_session") < visited.index("recommend_videos")


def test_after_each_answer_tracker_and_progress_run_without_a_match(runner, learner, vocab_dir, visited):
    tid = runner.start_session(learner, TOPIC)["thread_id"]
    pass_pretest(runner, tid, ["B", *ANSWERS[1:]])
    runner.resume(tid, {"query": "How do plants make glucose from light?"})
    values = runner.get_state(tid)["values"]
    assert visited[-4:] == ["generate_response", "update_tracker", "monitor_progress", "coach"]
    # The question names no concept and the fake services have no embedder, so match_question finds nothing.
    assert values["tracker_updates"][-1]["status"] == "no match"
    progress = values["progress"]
    assert progress["after_question"] == 1 and progress["touched"] == 0
    assert progress["suggestion"].startswith("Ask about Concept 1")


def test_matched_questions_are_recorded_once_and_move_progress(runner, learner, vocab_dir, monkeypatch):
    monkeypatch.setattr("matching.match_question", lambda text, topic_id: ("concept_1", 0.9))
    tid = runner.start_session(learner, TOPIC)["thread_id"]
    pass_pretest(runner, tid, ["B", *ANSWERS[1:]])
    runner.resume(tid, {"query": "How do plants make glucose from light?"})
    values = runner.get_state(tid)["values"]
    assert values["tracker_updates"][-1] == {"question_no": 1, "concept_id": "concept_1", "confidence": 0.9,
                                             "status": "recorded", "detail": ""}
    goal = next(g for g in values["progress"]["goals"] if g["concept_id"] == "concept_1")
    assert goal["asked"] == 1 and goal["status"] == "discussed"


def test_low_confidence_match_is_not_recorded(runner, learner, vocab_dir, monkeypatch):
    monkeypatch.setattr("matching.match_question", lambda text, topic_id: ("concept_1", 0.3))
    tid = runner.start_session(learner, TOPIC)["thread_id"]
    pass_pretest(runner, tid)
    runner.resume(tid, {"query": "How do plants make glucose from light?"})
    values = runner.get_state(tid)["values"]
    assert values["tracker_updates"][-1]["status"] == "low confidence"
    assert all(g["asked"] == 0 for g in values["progress"]["goals"])


def test_topic_without_vocabulary_runs_unplanned(runner, learner, recommend_calls):
    tid = runner.start_session(learner, TOPIC)["thread_id"]  # no fixture vocabulary for this topic
    out = pass_pretest(runner, tid)
    assert out["interrupt"]["type"] == "ask_question"
    values = runner.get_state(tid)["values"]
    assert values["session_plan"]["status"] == "unavailable" and values["session_plan"]["goals"] == []
    assert "concepts" not in recommend_calls[0]  # the recommender builds its own query, as before
    runner.resume(tid, {"query": "How do plants make glucose from light?"})
    assert runner.get_state(tid)["values"]["progress"] == {}


def test_plan_card_shows_goals_and_progress():
    from streamlit.testing.v1 import AppTest

    def card():
        from frontend.views.learning_session import _plan_card

        _plan_card({"session_plan": {"status": "ok", "query": {"concepts": ["Concept 1", "Concept 5"]}, "goals": [
            {"concept_id": "concept_1", "name": "Concept 1", "description": "What concept 1 is about.",
             "reason": "missed in this pre-test", "mastery": 0.33, "attempts": 1},
            {"concept_id": "concept_5", "name": "Concept 5", "description": "What concept 5 is about.",
             "reason": "low mastery 0.25 after 2 attempts", "mastery": 0.25, "attempts": 2}]},
            "progress": {"goals": [{"concept_id": "concept_1", "status": "discussed"},
                                   {"concept_id": "concept_5", "status": "not started"}],
                         "suggestion": "Ask about Concept 5 next."}})

    at = AppTest.from_function(card, default_timeout=30).run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Your goals for this session" in text and "**Concept 1** · missed in this pre-test" in text
    assert [c.value for c in at.caption][-1] == "Ask about Concept 5 next."
    badges = [m.value for m in at.markdown if "-badge[" in m.value]  # st.badge renders as a markdown badge
    assert len(badges) == 2 and "Discussed" in badges[0] and "Not started" in badges[1]
