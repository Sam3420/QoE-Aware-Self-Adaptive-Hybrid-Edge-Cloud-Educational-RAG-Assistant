"""Streamlit page (AppTest, headless): search → cards → feedback events; missing-key state. All YouTube calls mocked."""
from streamlit.testing.v1 import AppTest


def page_script(api_key: str | None) -> None:
    import tempfile

    import streamlit as st

    from agent.deps import make_fake_services, set_services
    from agent.recommendations import make_recommender
    from config import Settings
    from frontend import resources
    from frontend.views import recommendations
    from services.vector_store import HashingEmbedder
    from tests.youtube_fakes import FakeTranscripts, FakeYouTubeApi, make_client

    if "rec_env" not in st.session_state:  # build the fakes once per test session
        services = make_fake_services()
        set_services(services)
        cfg = Settings(_env_file=None, rec_similarity_floor=0.0, rec_similarity_ceil=0.6,
                       youtube_cache_dir=tempfile.mkdtemp(), transcript_chunk_words=20,
                       transcript_chunk_overlap_words=5, rec_concept_cover_threshold=0.5)
        rec = make_recommender(services, data_client=make_client(FakeYouTubeApi(), api_key=api_key),
                               transcripts=FakeTranscripts(), embedder=HashingEmbedder(), cfg=cfg)
        st.session_state["rec_env"] = (rec, services.app.create_learner("UI learner"))
    rec, learner = st.session_state["rec_env"]
    resources.get_recommender = lambda backend: rec

    class Backend:
        def topics(self):
            return {"photosynthesis": None}

    recommendations.render(Backend(), {"id": learner, "display_name": "UI learner"})


def test_search_shows_ranked_cards_and_records_feedback():
    at = AppTest.from_function(page_script, args=("test-key",), default_timeout=30).run()
    assert not at.exception
    at.text_input[0].input("photosynthesis chlorophyll light energy glucose")
    at.button[0].click().run()  # "Find videos" (form submit)
    assert not at.exception
    markdown = " ".join(m.value for m in at.markdown)
    assert "1. Photosynthesis explained simply" in markdown
    assert "Transcript (en, manual captions) addresses the objective" in markdown  # grounded explanation shown

    rec, _ = at.session_state["rec_env"]
    run = rec.get_run(at.session_state["rec_run"])
    helpful = next(b for b in at.button if b.key == f"acc-{run['recommendations'][0]['item_id']}")
    helpful.click().run()
    assert not at.exception
    from app.db.repositories import RecommendationRepository

    with rec.services.app._db() as db:
        types = [e["event_type"] for e in RecommendationRepository(db).list_events(run["id"])]
    assert "displayed" in types and "accepted" in types
    assert rec.get_run(run["id"])["reward"] is not None


def test_missing_api_key_disables_search():
    at = AppTest.from_function(page_script, args=(None,), default_timeout=30).run()
    assert not at.exception
    assert any("YOUTUBE_API_KEY" in w.value for w in at.warning)
    assert at.button[0].disabled
