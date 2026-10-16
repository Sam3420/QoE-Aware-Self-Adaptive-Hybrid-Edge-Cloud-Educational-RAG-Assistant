"""Topic selection → pre-test → video recommendation → video selection → knowledge preparation → Q&A."""
import pytest

from app.services.errors import MCQGenerationError
from services.fakes import FakeTranscriptFetcher, FakeVideoRecommender
from tests.conftest import ANSWERS, pick_video


def _to_video_choice(runner, learner, topic="life_processes"):
    out = runner.start_session(learner)
    assert out["interrupt"]["type"] == "select_topic"
    out = runner.resume(out["thread_id"], {"topic_id": topic})
    assert out["interrupt"]["type"] == "pre_test"
    out = runner.resume(out["thread_id"], {"answers": ANSWERS})
    assert out["interrupt"]["type"] == "select_video"
    return out


def test_session_without_topic_asks_for_a_catalogue_topic(runner, learner):
    out = runner.start_session(learner)
    pause = out["interrupt"]
    assert pause["type"] == "select_topic"
    assert [t["topic_id"] for t in pause["topics"]] == [
        "life_processes", "control_and_coordination", "how_do_organisms_reproduce", "heredity", "our_environment"]
    assert pause["topics"][0]["label"] == "Class 10 Biology · Life Processes"
    out = runner.resume(out["thread_id"], {"topic_id": "astrology"})
    assert out["interrupt"]["type"] == "select_topic" and "unknown topic" in out["interrupt"]["error"]
    out = runner.resume(out["thread_id"], {"topic_id": "how_do_organisms_reproduce"})
    assert out["interrupt"]["type"] == "pre_test"
    assert runner.get_state(out["thread_id"])["values"]["topic_id"] == "how_do_organisms_reproduce"


def test_chosen_video_is_indexed_and_answers_come_from_it(runner, learner, services, visited):
    out = _to_video_choice(runner, learner)
    tid, pause = out["thread_id"], out["interrupt"]
    rec = services.videos
    assert rec.calls == [{"student_id": learner, "topic_id": "life_processes", "session_id": tid, "revision": False}]
    assert pause["status"] == "ok" and len(pause["items"]) == 3 and pause["notice"] is None
    chosen = pause["items"][1]
    out = runner.resume(tid, {"item_id": chosen["item_id"]})
    assert out["interrupt"]["type"] == "ask_question"
    values = runner.get_state(tid)["values"]
    assert values["resource_id"] == f"yt_{chosen['video_id']}" and values["video"]["title"] == chosen["title"]
    assert values["video"]["reused_index"] is False and values["video"]["n_chunks"] > 0
    assert services.crag.store.built[f"yt_{chosen['video_id']}"]  # transcript chunks were embedded
    assert rec.transcripts.calls == [chosen["video_id"]]
    assert ("fake-run-1", "accepted", chosen["item_id"]) in rec.events
    runner.resume(tid, {"query": "How do plants convert sunlight into chemical energy?"})
    chunks = runner.get_state(tid)["values"]["chunks"]
    assert chunks and all(c["chunk_id"].startswith(f"yt_{chosen['video_id']}") for c in chunks)
    with services.app._db() as db:
        from app.db.models import EducationalResource

        stored = db.query(EducationalResource).filter_by(external_resource_id=chosen["video_id"]).one()
        assert stored.source == "youtube" and stored.resource_metadata["resource_id"] == values["resource_id"]
    assert visited.index("prepare_knowledge") < visited.index("collect_context")


def test_missing_transcript_falls_back_to_the_next_recommendation(runner, learner, services):
    services.videos = FakeVideoRecommender(transcripts=FakeTranscriptFetcher(missing=("fakeVid1000",)))
    out = _to_video_choice(runner, learner)
    out = runner.resume(out["thread_id"], {"item_id": out["interrupt"]["items"][0]["item_id"]})
    assert out["interrupt"]["type"] == "ask_question"
    values = runner.get_state(out["thread_id"])["values"]
    assert values["video"]["video_id"] == "fakeVid1001"
    assert values["video"]["replaces"]["video_id"] == "fakeVid1000"
    assert [a["status"] for a in values["knowledge_attempts"]] == ["failed", "ok"]
    assert any("skipped (transcript no_transcript" in e for e in values["session_errors"])


def test_no_usable_video_returns_to_selection_then_refresh_or_end(runner, learner, services):
    services.videos = FakeVideoRecommender(
        transcripts=FakeTranscriptFetcher(missing=("fakeVid1000", "fakeVid1001", "fakeVid1002")))
    out = _to_video_choice(runner, learner)
    tid = out["thread_id"]
    out = runner.resume(tid, {"item_id": out["interrupt"]["items"][2]["item_id"]})
    pause = out["interrupt"]
    assert pause["type"] == "select_video" and "None of the recommended videos" in pause["notice"]
    assert all(item["unavailable"] for item in pause["items"])
    out = runner.resume(tid, {"item_id": pause["items"][0]["item_id"]})  # unavailable: re-asked
    assert out["interrupt"]["type"] == "select_video" and "not one of the available" in out["interrupt"]["error"]

    out = runner.resume(tid, {"refresh": True})  # new run with new videos
    pause = out["interrupt"]
    assert pause["type"] == "select_video" and pause["notice"] is None
    assert [i["video_id"] for i in pause["items"]] == ["fakeVid2000", "fakeVid2001", "fakeVid2002"]
    out = runner.resume(tid, {"end": True})
    assert out["interrupt"]["type"] == "post_test_and_feedback"


def test_already_indexed_video_is_reused(runner, learner, services, visited):
    out = _to_video_choice(runner, learner)
    tid, item = out["thread_id"], out["interrupt"]["items"][0]
    services.crag.store.build(f"yt_{item['video_id']}", ["already indexed"])
    visited.clear()
    out = runner.resume(tid, {"item_id": item["item_id"]})
    assert out["interrupt"]["type"] == "ask_question"
    assert "prepare_knowledge" not in visited and services.videos.transcripts.calls == []
    assert runner.get_state(tid)["values"]["video"]["reused_index"] is True


def test_pretest_is_skipped_only_when_no_questions_can_be_prepared(runner, learner, services, visited):
    def fail(*a, **k):
        raise MCQGenerationError("no LLM")

    services.app.plan_pretest = fail
    out = runner.start_session(learner, "photosynthesis")
    assert out["interrupt"]["type"] == "select_video"
    assert "pre_test" not in visited and "recommend_videos" in visited
    assert any("pre_test skipped: no LLM" in e for e in runner.get_state(out["thread_id"])["values"]["session_errors"])


def test_real_coordinator_records_events_and_skips_unusable_transcripts(runner, learner, services, tmp_path):
    from agent.recommendations import make_recommender
    from config import Settings
    from services.vector_store import HashingEmbedder
    from tests.youtube_fakes import FakeTranscripts, FakeYouTubeApi, make_client

    cfg = Settings(_env_file=None, youtube_cache_dir=tmp_path, rec_similarity_floor=0.0, rec_similarity_ceil=0.6,
                   epsilon=0.0)
    transcripts = FakeTranscripts()
    services.videos = make_recommender(services, data_client=make_client(FakeYouTubeApi()), transcripts=transcripts,
                                       embedder=HashingEmbedder(), cfg=cfg)
    # The fake YouTube catalogue is about photosynthesis, so start with that topic preset (skips select_topic).
    out = runner.start_session(learner, "photosynthesis")
    out = runner.resume(out["thread_id"], {"answers": ANSWERS})
    pause = out["interrupt"]
    assert pause["type"] == "select_video"
    assert pause["status"] == "ok" and pause["strategy_id"] in {"R1", "R2", "R3"}
    ids = [i["video_id"] for i in pause["items"]]
    assert "goodTeach01" in ids
    transcripts.calls.clear()
    out = runner.resume(out["thread_id"], {"item_id": pause["items"][ids.index("goodTeach01")]["item_id"]})
    # The fake lesson transcript is 5 short lines: too short to teach from, so every candidate fails.
    pause = out["interrupt"]
    assert pause["type"] == "select_video" and "too short" in pause["notice"]
    assert transcripts.calls[0] == ("goodTeach01", ("en",))  # the learner's preferred language only
    run_id = runner.get_state(out["thread_id"])["values"]["recommendations"]["run_id"]
    with services.app._db() as db:
        from app.db.repositories import RecommendationRepository

        events = RecommendationRepository(db).list_events(run_id)
    assert "accepted" in [e["event_type"] for e in events]


@pytest.mark.parametrize("payload", [{}, {"item_id": "x", "end": True}, {"refresh": True, "end": True}])
def test_video_reply_needs_exactly_one_action(payload):
    from pydantic import ValidationError

    from agent.schemas import VideoReply

    with pytest.raises(ValidationError):
        VideoReply.model_validate(payload)


def test_pick_video_helper_matches_first_item(runner, learner):
    out = _to_video_choice(runner, learner)
    out = pick_video(runner, out["thread_id"], out)
    assert out["interrupt"]["type"] == "ask_question"


def test_after_doubts_the_learner_can_watch_another_video_then_take_the_post_test(runner, learner, services, visited):
    out = _to_video_choice(runner, learner)
    tid, first = out["thread_id"], out["interrupt"]["items"][0]
    out = runner.resume(tid, {"item_id": first["item_id"]})
    assert out["interrupt"]["video"]["title"] == first["title"]
    runner.resume(tid, {"query": "How do plants convert sunlight into chemical energy?"})  # a doubt on video 1
    policy = runner.get_state(tid)["values"]["policy_id"]

    out = runner.resume(tid, {"more_videos": True})  # back to the list, video 1 marked as watched
    pause = out["interrupt"]
    assert pause["type"] == "select_video" and [i["watched"] for i in pause["items"]] == [True, False, False]
    assert [v["video_id"] for v in pause["watched"]] == [first["video_id"]]

    visited.clear()
    second = pause["items"][1]
    out = runner.resume(tid, {"item_id": second["item_id"]})
    assert out["interrupt"]["type"] == "ask_question" and out["interrupt"]["video"]["title"] == second["title"]
    assert visited == ["select_video", "prepare_knowledge"]  # no new context measurement or policy selection
    runner.resume(tid, {"query": "What does chlorophyll absorb in the leaf?"})  # a doubt on video 2

    values = runner.get_state(tid)["values"]
    assert values["policy_id"] == policy and values["resource_id"] == f"yt_{second['video_id']}"
    assert [v["video_id"] for v in values["videos_watched"]] == [first["video_id"], second["video_id"]]
    assert [(e["resource_id"], e["video_title"]) for e in values["qa_log"]] == [
        (f"yt_{first['video_id']}", first["title"]), (f"yt_{second['video_id']}", second["title"])]
    assert ("fake-run-1", "accepted", second["item_id"]) in services.videos.events

    out = runner.resume(tid, {"end": True})
    assert out["interrupt"]["type"] == "post_test_and_feedback"
    final = runner.resume(tid, {"answers": ANSWERS, "rating": 5})["final_state"]
    assert len(final["qa_log"]) == 2 and final["evaluation"]["status"] == "completed"


def test_more_videos_and_end_cannot_both_be_sent(runner, learner):
    out = _to_video_choice(runner, learner)
    out = pick_video(runner, out["thread_id"], out)
    out = runner.resume(out["thread_id"], {"more_videos": True, "end": True})
    assert out["interrupt"]["type"] == "ask_question" and "not both" in out["interrupt"]["error"]


def test_top_recommendations_are_prefetched_while_the_learner_chooses(runner, learner, services, visited, monkeypatch):
    from agent import knowledge
    from config import settings

    monkeypatch.setattr(settings, "video_prefetch_count", 2)
    out = _to_video_choice(runner, learner)
    tid, items = out["thread_id"], out["interrupt"]["items"]
    knowledge._prefetcher.submit(lambda: None).result(timeout=10)  # the single worker has finished the queue
    assert sorted(services.videos.transcripts.calls) == sorted(i["video_id"] for i in items[:2])
    visited.clear()
    out = runner.resume(tid, {"item_id": items[0]["item_id"]})
    assert out["interrupt"]["type"] == "ask_question"
    assert "prepare_knowledge" not in visited  # already indexed in the background: no wait for the learner
    assert runner.get_state(tid)["values"]["resource_id"] == f"yt_{items[0]['video_id']}"
    with services.app._db() as db:
        from app.db.models import EducationalResource

        prefetched = db.query(EducationalResource).filter_by(external_resource_id=items[0]["video_id"]).one()
        assert prefetched.resource_metadata["resource_id"] == f"yt_{items[0]['video_id']}"
