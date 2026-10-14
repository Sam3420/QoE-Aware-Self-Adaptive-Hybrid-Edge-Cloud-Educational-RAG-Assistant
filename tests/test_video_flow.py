"""Topic selection → pre-test → video recommendation → video selection → knowledge preparation → Q&A."""
import pytest

from app.services.errors import MCQGenerationError
from services.fakes import FakeTranscriptFetcher, FakeVideoRecommender
from tests.conftest import ANSWERS, pick_video


def _to_video_choice(runner, learner, topic="photosynthesis"):
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
    assert "crop_production_and_management" in [t["topic_id"] for t in pause["topics"]]
    out = runner.resume(out["thread_id"], {"topic_id": "astrology"})
    assert out["interrupt"]["type"] == "select_topic" and "unknown topic" in out["interrupt"]["error"]
    out = runner.resume(out["thread_id"], {"topic_id": "reproduction_in_animals"})
    assert out["interrupt"]["type"] == "pre_test"
    assert runner.get_state(out["thread_id"])["values"]["topic_id"] == "reproduction_in_animals"


def test_chosen_video_is_indexed_and_answers_come_from_it(runner, learner, services, visited):
    out = _to_video_choice(runner, learner)
    tid, pause = out["thread_id"], out["interrupt"]
    rec = services.videos
    assert rec.calls == [{"student_id": learner, "topic_id": "photosynthesis", "session_id": tid, "revision": False}]
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
    out = _to_video_choice(runner, learner)
    pause = out["interrupt"]
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
