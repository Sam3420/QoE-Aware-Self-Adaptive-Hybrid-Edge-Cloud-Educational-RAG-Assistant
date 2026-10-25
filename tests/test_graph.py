"""End-to-end graph behaviour with fake services, B's real DB code (in-memory) and an in-memory checkpointer."""
from services.fakes import FakeLLMClient
from services.llm_clients import LLMError
from services.vector_store import RetrievedChunk
from tests.conftest import ANSWERS, seed_experiences, pass_pretest

SPECIFIC = "How do plants convert sunlight into chemical energy?"


class AlwaysWeakBackend:
    """Retrieval that never reaches the threshold, even after rewrites."""

    def search(self, resource_id, query, top_k):
        return [RetrievedChunk(chunk_id=f"w{i}", text="weak", score=0.4) for i in range(top_k)]

    def score_chunks(self, resource_id, query, chunk_ids):
        return {cid: 0.4 for cid in chunk_ids}


class DownClient(FakeLLMClient):
    def is_available(self):
        return False


class FailingClient(FakeLLMClient):
    def generate(self, request):
        raise LLMError("boom")


def _to_first_question(runner, learner):
    out = runner.start_session(learner, "photosynthesis")
    assert out["interrupt"]["type"] == "pre_test"
    out = pass_pretest(runner, out["thread_id"])
    assert out["interrupt"]["type"] == "ask_question"
    return out["thread_id"]


def _favour(services, policy_id):
    """Make kNN pick `policy_id` greedily."""
    # Same context as the session's: the learner aces the pre-test (advanced) and has no QoE history yet.
    ctx = {"rtt_ms": 180.0, "cpu": 0.35, "mem": 0.55, "competency_num": 1.0, "last_qoe": 0.5}
    seed_experiences(services, [(policy_id, 0.9), (policy_id, 0.9), ("P1", 0.1)], context=ctx)


def test_two_question_session_loops_and_finishes(runner, learner, services, visited):
    _favour(services, "P2")  # local, top_k 5, 1 retry
    tid = _to_first_question(runner, learner)
    runner.resume(tid, {"query": "photosynthesis?"})  # vague → 1 corrective retry
    runner.resume(tid, {"query": SPECIFIC})
    out = runner.resume(tid, {"end": True})
    assert out["interrupt"]["type"] == "post_test_and_feedback"
    out = runner.resume(tid, {"answers": ANSWERS, "rating": 4, "comment": "good"})
    assert visited == [
        "load_profile", "select_topic", "plan_pretest", "pre_test", "plan_session", "recommend_videos",
        "select_video", "prepare_knowledge", "collect_context", "select_policy",
        "ask_question", "crag_retrieve", "corrective_retrieve", "crag_retrieve", "llm_router", "generate_response",
        "update_tracker", "monitor_progress", "coach",
        "ask_question", "crag_retrieve", "llm_router", "generate_response", "update_tracker", "monitor_progress",
        "coach",
        "ask_question", "post_test_and_feedback", "reflect", "evaluate_session", "store_experience",
    ]
    final = out["final_state"]
    assert final["policy_id"] == "P2"
    assert [e["crag_retries"] for e in final["qa_log"]] == [1, 0]
    assert all(e["n_chunks"] == 5 for e in final["qa_log"])  # top_k from the policy
    assert final["pre_score"] == 1.0 and final["post_score"] == 1.0 and final["rating"] == 4


def test_low_score_triggers_exactly_max_retries(runner, learner, services, visited):
    services.crag.store = AlwaysWeakBackend()
    _favour(services, "P4")  # max_retries 1
    tid = _to_first_question(runner, learner)
    runner.resume(tid, {"query": SPECIFIC})
    assert visited.count("corrective_retrieve") == 1
    entry = runner.get_state(tid)["values"]["qa_log"][0]
    assert entry["crag_retries"] == 1 and entry["retrieval_score"] == 0.4


def test_zero_retry_policy_never_corrects(runner, learner, services, visited):
    services.crag.store = AlwaysWeakBackend()
    tid = _to_first_question(runner, learner)  # cold start → P1, max_retries 0
    runner.resume(tid, {"query": SPECIFIC})
    assert "corrective_retrieve" not in visited


def test_crag_retries_reset_for_each_question(runner, learner, services):
    services.crag.store = AlwaysWeakBackend()
    _favour(services, "P2")
    tid = _to_first_question(runner, learner)
    runner.resume(tid, {"query": SPECIFIC})
    runner.resume(tid, {"query": SPECIFIC})
    assert [e["crag_retries"] for e in runner.get_state(tid)["values"]["qa_log"]] == [1, 1]


def test_unavailable_route_falls_back_and_is_logged(runner, learner, services):
    services.router.clients["local"] = DownClient("local", "down", 1.0)
    tid = _to_first_question(runner, learner)  # P1 prefers local
    runner.resume(tid, {"query": SPECIFIC})
    entry = runner.get_state(tid)["values"]["qa_log"][0]
    assert entry["preferred_route"] == "local" and entry["model_used"].startswith("cloud:")
    assert "router: 'local' unavailable, used 'cloud'" in entry["error"]


def test_generation_failure_falls_back_to_other_route(runner, learner, services):
    services.router.clients["local"] = FailingClient("local", "flaky", 1.0)
    tid = _to_first_question(runner, learner)
    runner.resume(tid, {"query": SPECIFIC})
    entry = runner.get_state(tid)["values"]["qa_log"][0]
    assert entry["model_used"] == "cloud:cloud-llm-fake"
    assert "generation (local): boom" in entry["error"] and "answered by 'cloud'" in entry["error"]


def _finish(runner, tid, answers, rating=5):
    runner.resume(tid, {"end": True})
    return runner.resume(tid, {"answers": answers, "rating": rating})


def test_new_learner_gets_the_standard_pretest(runner, learner):
    out = runner.start_session(learner, "photosynthesis")
    pause = out["interrupt"]
    assert pause["type"] == "pre_test" and pause["plan"]["mode"] == "new"
    assert [i["item_id"] for i in pause["items"]] == [0, 1, 2, 3, 4]
    assert all(i["reason"] == "Starting quiz for this topic." for i in pause["items"])


def test_returning_learner_always_gets_a_pretest_planned_from_history(runner, learner, visited):
    # Session 1: questions 1 and 3 answered wrongly (correct letters are A, B, C, D, A).
    tid = runner.start_session(learner, "photosynthesis")["thread_id"]
    pass_pretest(runner, tid, ["B", "B", "A", "D", "A"])
    _finish(runner, tid, ["B", "B", "A", "D", "A"])  # post-test: same mistakes

    visited.clear()
    out = runner.start_session(learner, "photosynthesis")
    assert visited == ["load_profile", "select_topic", "plan_pretest"]  # pre_test is waiting for answers
    pause = out["interrupt"]
    plan = pause["plan"]
    assert pause["type"] == "pre_test" and plan["mode"] == "returning"
    assert plan["weak_concepts"] == ["concept 1", "concept 3"]
    assert plan["previous"]["pre_score"] == 0.6 and plan["previous"]["post_score"] == 0.6
    ids = [i["item_id"] for i in pause["items"]]
    # The two missed questions are re-checked; the rest are new questions generated into the bank.
    assert sorted(ids[:2]) == [0, 2]
    assert all(i >= 5 for i in ids[2:]) and len(ids) == 5
    assert all("similar question on the same idea" in i["reason"] for i in pause["items"][:2])
    assert not any("got this wrong" in i["reason"] for i in pause["items"])  # no claim about the exact question
    assert all(i["reason"].startswith("New question") for i in pause["items"][2:])


def test_post_test_reuses_the_sessions_pretest_questions(runner, learner):
    tid = runner.start_session(learner, "photosynthesis")["thread_id"]
    pass_pretest(runner, tid, ["B", "B", "A", "D", "A"])
    _finish(runner, tid, ["B", "B", "A", "D", "A"])

    tid2 = runner.start_session(learner, "photosynthesis")["thread_id"]
    pre_items = runner.get_state(tid2)["values"]["test_items"]
    pass_pretest(runner, tid2, ["A"] * 5)
    runner.resume(tid2, {"end": True})
    post = runner.get_state(tid2)["interrupt"]
    assert post["type"] == "post_test_and_feedback"
    assert [i["question"] for i in post["items"]] == [i["question"] for i in pre_items]


def test_invalid_payloads_are_reasked(runner, learner):
    tid = runner.start_session(learner, "photosynthesis")["thread_id"]
    out = runner.resume(tid, {"answers": "ABCDA"})  # not a list
    assert out["interrupt"]["type"] == "pre_test" and "error" in out["interrupt"]
    pass_pretest(runner, tid, ANSWERS)
    out = runner.resume(tid, {"query": "   "})
    assert out["interrupt"]["type"] == "ask_question" and "error" in out["interrupt"]
    out = runner.resume(tid, {"end": True})
    out = runner.resume(tid, {"answers": ANSWERS, "rating": 9})
    assert out["interrupt"]["type"] == "post_test_and_feedback" and "error" in out["interrupt"]


def test_scores_are_shown_and_answers_revealed_only_after_the_post_test(runner, learner):
    tid = runner.start_session(learner, "photosynthesis")["thread_id"]
    pass_pretest(runner, tid, ["A", "A", "A", "A", "A"])  # fake key is A, B, C, D, A: 2 right
    values = runner.get_state(tid)["values"]
    assert values["pre_result"] == {"correct": 2, "total": 5, "competency": values["competency"],
                                    "answers": ["A"] * 5}
    assert values["post_review"] == {}  # nothing revealed before the post-test
    runner.resume(tid, {"end": True})
    runner.resume(tid, {"answers": ["A", "B", "C", "D", "B"], "rating": 4})
    review = runner.get_state(tid)["values"]["post_review"]
    assert (review["correct"], review["total"], review["note"]) == (4, 5, None)
    first, last = review["items"][0], review["items"][4]
    assert first["answer"] == "A" and first["post_correct"] and first["pre_correct"] and first["explanation"]
    assert last["answer"] == "A" and last["post_answer"] == "B" and not last["post_correct"] and last["pre_correct"]
