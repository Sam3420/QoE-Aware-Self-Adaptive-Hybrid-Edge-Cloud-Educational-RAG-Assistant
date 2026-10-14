"""Decision layer + persistence: strategy selection, runs, events, outcomes, reward (in-memory DB, mocked YouTube)."""
import pytest

from agent.recommendations import STRATEGIES, make_recommender
from app.db.errors import ReferencedRecordNotFoundError
from app.db.repositories import RecommendationRepository
from config import Settings
from services.vector_store import HashingEmbedder
from services.youtube.metrics import qoe_signals, recommendation_metrics, reward_stub
from tests.conftest import ANSWERS
from tests.youtube_fakes import OBJECTIVE, FakeTranscripts, FakeYouTubeApi, api_error, make_client


@pytest.fixture
def cfg(tmp_path):
    return Settings(_env_file=None, rec_similarity_floor=0.0, rec_similarity_ceil=0.6, youtube_cache_dir=tmp_path,
                    transcript_chunk_words=20, transcript_chunk_overlap_words=5, rec_concept_cover_threshold=0.5,
                    epsilon=0.0)


@pytest.fixture
def api():
    return FakeYouTubeApi()


@pytest.fixture
def rec(services, cfg, api):
    return make_recommender(services, data_client=make_client(api), transcripts=FakeTranscripts(),
                            embedder=HashingEmbedder(), cfg=cfg)


def _items(run):
    return {i["video_id"]: i for i in run["recommendations"]}


def test_recommend_stores_run_items_and_generated_events(rec, learner, services):
    run = rec.recommend(learner, "photosynthesis", objective=OBJECTIVE, concepts=["chlorophyll"])
    assert run["status"] == "ok" and run["strategy_id"] == "R2"
    assert run["decision_reason"].startswith("cold start") and "→ default R2" in run["decision_reason"]
    assert set(run["context"]) == {"rtt_ms", "cpu", "mem", "competency_num", "last_qoe"}
    assert list(_items(run)) == ["goodTeach01", "popularNoTr"] and all(r["item_id"] for r in run["recommendations"])
    assert run["metrics"]["rates"]["transcript_success_rate"] == 0.5 and run["metrics"]["quota_units"] == 101
    with services.app._db() as db:
        events = RecommendationRepository(db).list_events(run["id"])
    assert [e["event_type"] for e in events] == ["generated", "generated"]


def test_gap_concepts_come_from_wrong_pretest_answers(rec, learner, services):
    session = services.app.create_learning_session(learner)
    services.app.prepare_pretest(learner, "photosynthesis")
    services.app.submit_pretest(learner, session, "photosynthesis", ["A", "A", "A", "A", "A"])  # 2 of 5 correct
    needs = rec.learner_needs(learner, "photosynthesis")
    assert needs["gap_concepts"] == ["concept 2", "concept 3", "concept 4"]
    assert needs["competency"] == "intermediate" and needs["competency_num"] == 0.5
    assert needs["language_code"] == "en"  # learner default "english"
    run = rec.recommend(learner, "photosynthesis")
    assert run["objective"] == "Photosynthesis class 7: concept 2, concept 3, concept 4"  # catalogue title, not the id
    assert run["request"]["concepts"] == ["concept 2", "concept 3", "concept 4"]


def test_events_produce_outcomes_and_placeholder_reward(rec, learner):
    run = rec.recommend(learner, "photosynthesis", objective=OBJECTIVE)
    good, popular = _items(run)["goodTeach01"]["item_id"], _items(run)["popularNoTr"]["item_id"]
    out = rec.record_event(run["id"], "displayed", item_id=good)["outcome"]
    assert out["reward"] is None  # no explicit feedback yet: the run cannot influence selection
    rec.record_event(run["id"], "displayed", item_id=popular)
    rec.record_event(run["id"], "opened", item_id=good)
    rec.record_event(run["id"], "accepted", item_id=good)
    out = rec.record_event(run["id"], "rated", item_id=good, value={"rating": 4})["outcome"]
    m = out["metrics"]
    assert m["click_through_rate"]["value"] == 0.5 and m["acceptance_rate"]["value"] == 0.5
    assert m["learner_satisfaction"]["value"] == 0.8 and m["completion_rate"]["value"] == 0.0
    assert m["learning_gain"]["available"] is False  # no assessments around the run
    assert out["qoe_signals"]["UX"]["acceptance_rate"] == 0.5 and out["reward"] is not None
    assert out["reward_is_placeholder"]
    assert rec.get_run(run["id"])["reward"] == out["reward"]


def test_dismissed_videos_are_not_recommended_again_unless_revising(rec, learner):
    run = rec.recommend(learner, "photosynthesis", objective=OBJECTIVE)
    rec.record_event(run["id"], "dismissed", item_id=_items(run)["goodTeach01"]["item_id"])
    again = rec.recommend(learner, "photosynthesis", objective=OBJECTIVE)
    assert "goodTeach01" not in _items(again)
    assert "goodTeach01" in _items(rec.recommend(learner, "photosynthesis", objective=OBJECTIVE, revision=True))


def test_knn_selects_strategy_with_best_outcomes(rec, learner, services):
    with services.app._db() as db:
        repo = RecommendationRepository(db)
        ctx = {"rtt_ms": 180.0, "cpu": 0.35, "mem": 0.55, "competency_num": 0.0, "last_qoe": 0.5}
        for sid, reward in [("R3", 0.9), ("R3", 0.8), ("R1", 0.2), ("R2", 0.3)]:
            r = repo.create_run(student_id=learner, session_id=None, topic_id="photosynthesis", objective="x", query="x",
                                strategy_id=sid, decision_reason="seed", context=ctx, strategy={}, request={},
                                status="ok", errors=[], metrics={}, items=[])
            repo.update_outcome(r["id"], outcome={}, qoe_signals={}, reward=reward)
    run = rec.recommend(learner, "photosynthesis", objective=OBJECTIVE)
    assert run["strategy_id"] == "R3" and run["decision_reason"].startswith("R3: mean reward 0.85")
    assert run["strategy"]["transcript_budget"] == STRATEGIES["R3"].transcript_budget


def test_learning_gain_uses_assessments_around_the_run(rec, learner, services):
    session = services.app.create_learning_session(learner)
    services.app.prepare_pretest(learner, "photosynthesis")
    services.app.submit_pretest(learner, session, "photosynthesis", ["A", "A", "A", "A", "A"])  # 0.4
    run = rec.recommend(learner, "photosynthesis", objective=OBJECTIVE, session_id=session)
    services.app.submit_posttest_and_feedback(learner, session, "photosynthesis", ANSWERS, 5, None)  # 1.0
    m = rec.evaluate(run["id"])["metrics"]
    assert m["learning_gain"]["value"] == 0.6 and m["normalized_learning_gain"]["value"] == 1.0
    assert m["normalized_post_test"]["value"] == 1.0 and "not attributable" in m["learning_gain"]["note"]


def test_invalid_requests_and_events(rec, learner):
    with pytest.raises(ValueError):
        rec.recommend(learner, "photosynthesis", strategy_id="R9")
    run = rec.recommend(learner, "photosynthesis", objective=OBJECTIVE)
    item = run["recommendations"][0]["item_id"]
    for kwargs in ({"event_type": "rated", "item_id": item, "value": {"rating": 7}},
                   {"event_type": "rated", "item_id": item, "value": None},
                   {"event_type": "accepted", "item_id": None},
                   {"event_type": "watched_fully", "item_id": item}):
        with pytest.raises(ValueError):
            rec.record_event(run["id"], **kwargs)
    other = rec.recommend(learner, "photosynthesis", objective=OBJECTIVE, revision=True)
    with pytest.raises(ReferencedRecordNotFoundError):
        rec.record_event(run["id"], "accepted", item_id=other["recommendations"][0]["item_id"])


def test_failed_search_is_stored_as_failed_run(rec, learner, api):
    api.fail["search"] = api_error(403, "quotaExceeded")
    run = rec.recommend(learner, "photosynthesis", objective=OBJECTIVE)
    assert run["status"] == "failed" and run["recommendations"] == []
    assert any("quota_exceeded" in e for e in run["errors"])
    assert run["metrics"]["rates"]["api_failure_rate"] == 1.0


# ---- pure metric definitions -----------------------------------------------------------------------

def test_metrics_are_unavailable_without_evidence():
    m = recommendation_metrics([{"id": "a", "rank": 1}], [], None, None)
    assert all(not v["available"] for v in m.values())
    signals = qoe_signals(m, {"timings": {"total_ms": 3000}, "rates": {"api_failure_rate": 0.0}},
                          latency_budget_ms=15000, rtt_ms=None, rtt_max_ms=2000)
    assert signals["AQ"]["latency_quality"] == 0.8 and signals["NQ"]["rtt_quality"] is None
    assert signals["UX"]["feedback_sentiment"] is None and reward_stub(signals, {"judged": 0, "rated": 0}) is None


def test_precision_at_k_uses_explicit_labels_only():
    items = [{"id": "a", "rank": 1}, {"id": "b", "rank": 2}, {"id": "c", "rank": 3}]
    events = [{"event_type": "accepted", "item_id": "a"}, {"event_type": "dismissed", "item_id": "b"},
              {"event_type": "dismissed", "item_id": "a"}]  # accept wins over dismiss
    m = recommendation_metrics(items, events, None, None, k=3)
    assert m["precision_at_k"]["value"] == 0.5 and m["precision_at_k"]["denominator"] == 2  # c unlabelled
    assert m["acceptance_rate"]["denominator"] == 2  # interacted items count as shown


def test_rating_without_a_click_is_not_counted_as_a_dismissal(rec, learner):
    run = rec.recommend(learner, "photosynthesis", objective=OBJECTIVE)
    item = run["recommendations"][0]["item_id"]
    out = rec.record_event(run["id"], "rated", item_id=item, value={"rating": 5})["outcome"]
    assert out["metrics"]["acceptance_rate"]["value"] == 0.0  # the metric itself stays as defined
    latency = out["qoe_signals"]["AQ"]["latency_quality"]
    assert out["reward"] == round(0.8 * 1.0 + 0.2 * latency, 4)  # reward uses satisfaction only
