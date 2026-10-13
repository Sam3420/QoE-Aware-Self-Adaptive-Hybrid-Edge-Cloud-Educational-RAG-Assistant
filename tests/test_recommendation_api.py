"""HTTP API for recommendations (FastAPI TestClient; recommender overridden with fakes, no YouTube calls)."""
import pytest
from fastapi.testclient import TestClient

from agent.recommendations import make_recommender
from app.api.routes.recommendations import get_recommender
from app.main import app
from config import Settings
from services.vector_store import HashingEmbedder
from tests.youtube_fakes import OBJECTIVE, FakeTranscripts, FakeYouTubeApi, make_client


@pytest.fixture
def client(services, tmp_path):
    cfg = Settings(_env_file=None, rec_similarity_floor=0.0, rec_similarity_ceil=0.6, youtube_cache_dir=tmp_path,
                   transcript_chunk_words=20, transcript_chunk_overlap_words=5, rec_concept_cover_threshold=0.5)
    rec = make_recommender(services, data_client=make_client(FakeYouTubeApi()), transcripts=FakeTranscripts(),
                           embedder=HashingEmbedder(), cfg=cfg)
    app.dependency_overrides[get_recommender] = lambda: rec
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_recommend_event_metrics_round_trip(client, learner):
    resp = client.post(f"/learners/{learner}/recommendations",
                       json={"topic_id": "photosynthesis", "objective": OBJECTIVE, "concepts": ["chlorophyll"]})
    assert resp.status_code == 200
    run = resp.json()
    top = run["recommendations"][0]
    assert top["video_id"] == "goodTeach01" and top["url"] == "https://www.youtube.com/watch?v=goodTeach01"
    assert top["evidence_level"] == "transcript" and top["transcript_evidence"]["excerpts"]
    assert set(top["score_breakdown"]) >= {"topic_similarity", "transcript_relevance", "popularity"}

    ev = client.post(f"/recommendations/{run['id']}/events",
                     json={"event_type": "rated", "item_id": top["item_id"], "value": {"rating": 5}})
    assert ev.status_code == 201 and ev.json()["outcome"]["metrics"]["learner_satisfaction"]["value"] == 1.0

    assert client.get(f"/recommendations/{run['id']}").json()["reward"] is not None
    assert client.get(f"/recommendations/{run['id']}/metrics").status_code == 200
    assert [r["id"] for r in client.get(f"/learners/{learner}/recommendations").json()] == [run["id"]]


def test_api_errors(client, learner):
    assert client.post("/learners/nobody/recommendations", json={"topic_id": "t"}).status_code == 404
    assert client.post(f"/learners/{learner}/recommendations", json={"topic_id": ""}).status_code == 422
    assert client.get("/recommendations/missing").status_code == 404
    run = client.post(f"/learners/{learner}/recommendations", json={"topic_id": "photosynthesis",
                                                                     "objective": OBJECTIVE}).json()
    bad = client.post(f"/recommendations/{run['id']}/events", json={"event_type": "rated",
                                                                     "item_id": run["recommendations"][0]["item_id"],
                                                                     "value": {"rating": 9}})
    assert bad.status_code == 422
    assert client.post(f"/recommendations/{run['id']}/events",
                       json={"event_type": "accepted", "item_id": "nope"}).status_code == 404
