"""Day 3: scaffolding, the reflect node and its metrics, coach_mode off vs rules, the coach_data stub switch and
video coverage on the cards (fake services, fixture vocabulary; matching/formative_pool are stubs)."""
from datetime import datetime, timezone

import pytest

import knowledge_prep.concepts as concepts_mod
from agent import coach_data
from agent.coach import choose_scaffold
from agent.reflection import build_reflection
from config import settings
from knowledge_prep.concepts import Vocabulary, save_vocabulary
from services.llm_clients import build_request
from tests.conftest import ANSWERS, pass_pretest

# Formative checks serve fixed items here; the real pool is tested in test_coach_data_contract.py.
pytestmark = pytest.mark.usefixtures("fixed_formative_items")

TOPIC = "photosynthesis"
MISS_1 = ["B", *ANSWERS[1:]]  # pre-test: item 1 ("concept 1") wrong


# --- scaffolding -------------------------------------------------------------------------------------------------------

def test_choose_scaffold_rules():
    two_misses = {"c": {"mastery": 0.3, "formative_misses": 2}}
    scaffold = choose_scaffold(two_misses, "c", {"explanation_style": "short"})  # misses beat the profile
    assert (scaffold["style"], scaffold["support"]) == ("step_by_step", "hint_first")
    one_miss = {"c": {"mastery": 0.4, "formative_misses": 1}}
    assert choose_scaffold(one_miss, "c", {})["style"] == "concise"
    assert choose_scaffold(two_misses, None, {})["support"] == "direct"  # question's concept unknown
    by_profile = choose_scaffold({}, None, {"explanation_style": "Step by step, give me hints first"})
    assert (by_profile["style"], by_profile["support"]) == ("step_by_step", "hint_first")
    assert {k: choose_scaffold({}, None, {})[k] for k in ("style", "support")} == {"style": "concise",
                                                                                  "support": "direct"}


def test_scaffold_is_prompt_instructions_only():
    plain = build_request("Why are leaves green?", ["Chlorophyll is green."], grounded=True, competency="beginner",
                          language="en", max_tokens=200, temperature=0.2)
    shaped = build_request("Why are leaves green?", ["Chlorophyll is green."], grounded=True, competency="beginner",
                           language="en", max_tokens=200, temperature=0.2,
                           scaffold={"style": "step_by_step", "support": "hint_first"})
    assert "Hint:" in shaped.prompt and "Think about it:" in shaped.prompt and "numbered steps" in shaped.prompt
    assert "Hint:" not in plain.prompt and shaped.max_tokens == 300 and plain.max_tokens == 200
    assert shaped.system == plain.system  # same competency style; one request, no extra call


# --- reflection --------------------------------------------------------------------------------------------------------

def test_build_reflection():
    goals = [{"concept_id": "a", "name": "Alpha", "mastery": 1 / 3},
             {"concept_id": "b", "name": "Beta", "mastery": 0.5}]
    before = {"a": {"mastery": 1 / 3, "attempts": 1}, "b": {"mastery": 0.5, "attempts": 0},
              "c": {"mastery": 0.5, "attempts": 0}}
    after = {"a": {"mastery": 0.6, "attempts": 4}, "b": {"mastery": 0.5, "attempts": 0},
             "c": {"mastery": 0.25, "attempts": 2}}
    log = [{"action": "formative_check", "concept_id": "a", "accepted": True},
           {"action": "follow_up", "concept_id": "b", "accepted": False}]
    r = build_reflection(goals, before, after, {"a": "Alpha", "b": "Beta", "c": "Gamma"}, log, 1, "rules")
    assert [(g["concept_id"], g["met"]) for g in r["goals"]] == [("a", True), ("b", False)]  # a: +0.27 gain
    assert r["goal_completion_rate"] == 0.5 and r["suggestion_acceptance_rate"] == 0.5
    assert r["mean_mastery_gain"] == pytest.approx((0.2667 - 0.25) / 2, abs=1e-3)  # practised: a and c
    assert r["weak_next_time"] == ["c", "b"]
    # agent/reflection_template.py: goals met (named), the biggest change, the slip, and what to do next
    assert r["summary"].startswith("You met 1 of your 2 goals (**Alpha**).") and "**Gamma** slipped" in r["summary"]
    assert "Next time, start with **Gamma**" in r["summary"]


# --- graph -------------------------------------------------------------------------------------------------------------

@pytest.fixture
def video_covers_all(monkeypatch):
    """The chosen video covers every concept, so rule 2 (another_video) stays out of the way."""
    monkeypatch.setattr("matching.video_coverage", lambda video_id, topic_id: [f"concept_{i}" for i in range(1, 7)])


@pytest.fixture
def vocab_dir(tmp_path, monkeypatch):
    concepts = [{"id": f"concept_{i}", "name": f"Concept {i}", "description": f"What concept {i} is about.",
                 "aliases": [], "evidence": "Plants make food from light."} for i in range(1, 7)]
    monkeypatch.setattr(concepts_mod, "CONCEPTS_DIR", tmp_path)
    save_vocabulary(Vocabulary(topic_id=TOPIC, generated_by="test",
                               generated_at=datetime(2026, 10, 24, tzinfo=timezone.utc), concepts=concepts))


@pytest.fixture
def matched(monkeypatch):
    monkeypatch.setattr("matching.match_question", lambda text, topic_id: ("concept_1", 0.9))


@pytest.fixture
def prompts(services):
    seen, original = [], services.router.generate

    def spy(provider, request):
        seen.append(request)
        return original(provider, request)

    services.router.generate = spy
    return seen


def _finish(runner, tid):
    runner.resume(tid, {"end": True})
    return runner.resume(tid, {"answers": ANSWERS, "rating": 4, "comment": "good"})["final_state"]


def test_two_practice_misses_switch_answers_to_hint_first(runner, learner, vocab_dir, matched, prompts):
    tid = runner.start_session(learner, TOPIC)["thread_id"]
    pass_pretest(runner, tid, MISS_1)
    runner.resume(tid, {"query": "How do plants make glucose?"})
    assert runner.get_state(tid)["values"]["qa_log"][-1]["scaffold"]["support"] == "direct"
    runner.resume(tid, {"accept_suggestion": True})  # formative_check on concept_1
    runner.resume(tid, {"answer": "B"})  # wrong
    runner.resume(tid, {"answer": "A"})  # wrong
    runner.resume(tid, {"query": "Explain glucose making again."})
    entry = runner.get_state(tid)["values"]["qa_log"][-1]
    assert entry["concept_id"] == "concept_1"
    assert {k: entry["scaffold"][k] for k in ("style", "support")} == {"style": "step_by_step",
                                                                        "support": "hint_first"}
    assert "Hint:" in prompts[-1].prompt and "Hint:" not in prompts[0].prompt


def test_rules_session_reflects_saves_metrics_and_carries_weak_concepts_over(runner, learner, services, vocab_dir,
                                                                             matched):
    tid = runner.start_session(learner, TOPIC)["thread_id"]
    pass_pretest(runner, tid, MISS_1)
    runner.resume(tid, {"query": "How do plants make glucose?"})
    runner.resume(tid, {"accept_suggestion": True})
    runner.resume(tid, {"answer": "A"})
    runner.resume(tid, {"answer": "B"})  # both practice answers right
    final = _finish(runner, tid)
    reflection = final["reflection"]
    assert reflection["status"] == "ok" and reflection["coach_mode"] == "rules"
    assert reflection["formative_checks"] == 1 and reflection["suggestions_accepted"] == 1
    goal_1 = next(g for g in reflection["goals"] if g["concept_id"] == "concept_1")
    assert goal_1["before"] == pytest.approx(1 / 3, abs=1e-3) and goal_1["after"] > goal_1["before"]
    ev = final["evaluation"]
    assert ev["coach_mode"] == "rules" and ev["reflection"]["summary"]
    assert {"MG", "GC", "SA"} <= set(ev["metrics"]) and ev["metrics"]["SA"] is not None
    coach = ev["next_state"]["coach"]  # saved with the transition (store_experience)
    assert coach["coach_mode"] == "rules" and "concept_1" in coach["mastery_gain_by_concept"]
    weak = services.app.get_learner_profile(learner)["weak_concepts"][TOPIC]
    assert weak == reflection["weak_next_time"] and weak

    # next session: a concept left weak is planned even though the pre-test is now all right
    tid2 = runner.start_session(learner, TOPIC)["thread_id"]
    pass_pretest(runner, tid2, ANSWERS)
    goals = runner.get_state(tid2)["values"]["session_plan"]["goals"]
    assert goals[0]["concept_id"] == weak[0] and goals[0]["reason"] == "still weak after your last session"


def test_off_mode_is_the_reactive_system_but_still_logs_mastery(runner, learner, vocab_dir, matched, monkeypatch,
                                                                prompts):
    monkeypatch.setattr(settings, "coach_mode", "off")
    tid = runner.start_session(learner, TOPIC)["thread_id"]
    out = pass_pretest(runner, tid, MISS_1)
    values = runner.get_state(tid)["values"]
    assert values["coach_mode"] == "off" and values["session_plan"] == {"status": "off", "goals": [], "query": {}}
    assert values["mastery_start"]["concept_1"]["attempts"] == 1  # mastery logged at the start all the same
    out = runner.resume(tid, {"query": "How do plants make glucose?"})
    assert out["interrupt"]["suggestion"] is None
    values = runner.get_state(tid)["values"]
    assert values["tracker_updates"][-1]["status"] == "recorded"  # update_tracker runs in both modes
    assert values["qa_log"][-1]["scaffold"] is None and "Hint:" not in prompts[-1].prompt
    out = runner.resume(tid, {"accept_suggestion": True})
    assert "no suggestion" in out["interrupt"]["error"]  # nothing to accept in off mode
    final = _finish(runner, tid)
    assert final["reflection"]["coach_mode"] == "off" and final["reflection"]["goals"] == []
    assert final["evaluation"]["coach_mode"] == "off" and final["evaluation"]["coach"]["shown"] == 0
    assert final["evaluation"]["next_state"]["coach"]["coach_mode"] == "off"


def test_coach_mode_is_fixed_when_the_session_starts(runner, learner, vocab_dir, monkeypatch):
    tid = runner.start_session(learner, TOPIC)["thread_id"]  # rules
    monkeypatch.setattr(settings, "coach_mode", "off")
    pass_pretest(runner, tid, MISS_1)
    assert runner.get_state(tid)["values"]["session_plan"]["status"] == "ok"


def test_stub_switch_never_calls_the_real_functions(monkeypatch):
    calls = []
    monkeypatch.setattr("matching.match_question", lambda *a: calls.append(a) or ("concept_1", 0.9))
    monkeypatch.setattr(settings, "coach_data_source", "stub")
    assert coach_data.match_question("q", TOPIC) == (None, 0.0) and calls == []
    assert coach_data.video_coverage("v", TOPIC) is None
    assert [i.item_id for i in coach_data.get_formative_items(TOPIC, "concept_1", 2, [])] == [
        "stub:concept_1:1", "stub:concept_1:2"]
    monkeypatch.setattr(settings, "coach_data_source", "real")
    assert coach_data.match_question("q", TOPIC) == ("concept_1", 0.9) and len(calls) == 1


def test_video_cards_show_coverage_in_rules_mode_only(runner, learner, vocab_dir, monkeypatch):
    monkeypatch.setattr("matching.video_coverage", lambda video_id, topic_id: ["concept_1", "concept_3"])
    tid = runner.start_session(learner, TOPIC)["thread_id"]
    out = runner.resume(tid, {"answers": MISS_1})
    assert out["interrupt"]["type"] == "select_video"
    assert out["interrupt"]["items"][0]["covers"] == ["Concept 1", "Concept 3"]
    monkeypatch.setattr(settings, "coach_mode", "off")
    tid = runner.start_session(learner, TOPIC)["thread_id"]
    out = runner.resume(tid, {"answers": MISS_1})
    assert "covers" not in out["interrupt"]["items"][0]


def test_demo_path_on_stubs_follow_up_then_formative_check(runner, learner, vocab_dir, video_covers_all):
    """With the stub matcher, accepting a follow_up tags its question with the goal concept, so a weak goal gets
    a formative check next: the demo path works before matching.py is implemented."""
    tid = runner.start_session(learner, TOPIC)["thread_id"]
    pass_pretest(runner, tid, MISS_1)
    out = runner.resume(tid, {"query": "How do plants make glucose?"})
    assert out["interrupt"]["suggestion"]["action"] == "follow_up"
    out = runner.resume(tid, {"accept_suggestion": True})
    assert runner.get_state(tid)["values"]["tracker_updates"][-1]["status"] == "recorded"
    suggestion = out["interrupt"]["suggestion"]
    assert suggestion["action"] == "formative_check" and suggestion["concept_id"] == "concept_1"


def test_coverage_unknown_at_recommendation_is_rechecked_once_the_video_is_indexed(runner, learner, vocab_dir,
                                                                                    monkeypatch):
    """Live bug: coverage was "unknown" when the videos were recommended (no transcript index yet) and that unknown
    was cached for the session, so rule 2 (another_video) never fired for the chosen video."""
    import matching
    from tests.conftest import pick_video

    indexed = {"done": False}

    def coverage(video_id, topic_id):
        if not indexed["done"]:
            raise matching.CoverageUnknownError(f"No transcript index 'yt_{video_id}' yet.")
        return ["concept_1"]  # the chosen video covers only concept_1

    monkeypatch.setattr("matching.video_coverage", coverage)
    tid = runner.start_session(learner, TOPIC)["thread_id"]
    out = runner.resume(tid, {"answers": MISS_1})
    assert out["interrupt"]["type"] == "select_video"
    assert all(item["covers"] is None for item in out["interrupt"]["items"])  # unknown: no line on the cards
    assert runner.get_state(tid)["values"]["video_concepts"] == {}  # and unknown is not cached

    indexed["done"] = True  # prepare_knowledge indexes the chosen video
    pick_video(runner, tid, out)
    out = runner.resume(tid, {"query": "Why are leaves green?"})  # names no concept: rule 1 does not apply
    suggestion = out["interrupt"]["suggestion"]
    goals = [g["concept_id"] for g in runner.get_state(tid)["values"]["session_plan"]["goals"]]
    uncovered = next(g for g in goals if g != "concept_1")
    assert (suggestion["action"], suggestion["concept_id"]) == ("another_video", uncovered)
    video_id = runner.get_state(tid)["values"]["video"]["video_id"]
    assert runner.get_state(tid)["values"]["video_concepts"][video_id] == ["concept_1"]  # known: cached now
