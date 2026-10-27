"""Ranking features, missing-data rules, hard filters and the end-to-end pipeline on a synthetic catalogue."""
import pytest
from pydantic import ValidationError

from agent.recommendations import STRATEGIES
from services.vector_store import EmbeddingError, HashingEmbedder
from services.youtube.models import TranscriptRecord, TranscriptStatus, VideoCandidate
from services.youtube.pipeline import RecommendationRequest, RecommendationStrategy, build_query
from services.youtube.ranking import (
    RankingWeights,
    caption_match,
    combine,
    difficulty_match,
    duration_match,
    estimate_video_level,
    hard_filter,
    historical_outcome,
    language_match,
    popularity_scores,
)
from tests.youtube_fakes import OBJECTIVE, FakeTranscripts, FakeYouTubeApi, api_error, make_pipeline

W = RankingWeights.from_mapping({"topic_similarity": 1, "transcript_relevance": 1})


def request(**kw):
    base = dict(topic="photosynthesis", objective=OBJECTIVE, concepts=["chlorophyll", "glucose"], language_code="en",
                competency="beginner", competency_num=0.0)
    return RecommendationRequest(**{**base, **kw})


# ---- weights and score combination ----------------------------------------------------------------

def test_weights_are_validated_and_normalised():
    assert RankingWeights.from_mapping({"topic_similarity": 3, "popularity": 1}).weights["topic_similarity"] == 0.75
    for bad in ({"views": 1}, {"popularity": -1}, {"popularity": 0}, {"popularity": float("nan")}):
        with pytest.raises(ValidationError):
            RankingWeights.from_mapping(bad)
    with pytest.raises(ValidationError):
        RecommendationStrategy(strategy_id="X", name="x", max_candidates=5, transcript_budget=0, weights={"bogus": 1})


def test_missing_features_use_neutral_prior_not_zero():
    assert combine({"topic_similarity": 0.9, "transcript_relevance": None}, W, 0.5) == (0.7, 0.5)
    assert combine({"topic_similarity": 0.9, "transcript_relevance": 0.9}, W, 0.5) == (0.9, 1.0)
    assert combine({"topic_similarity": None, "transcript_relevance": None}, W, 0.5) == (0.5, 0.0)
    # a missing transcript is never worse than a confirmed irrelevant one
    assert combine({"topic_similarity": 0.9, "transcript_relevance": None}, W, 0.5)[0] > \
        combine({"topic_similarity": 0.9, "transcript_relevance": 0.0}, W, 0.5)[0]


# ---- features -----------------------------------------------------------------------------------

def test_duration_language_caption_and_difficulty_features():
    assert duration_match(600, 240, 1200) == 1.0 and duration_match(120, 240, 1200) == 0.5
    assert duration_match(1800, 240, 1200) == 0.5 and duration_match(None, 240, 1200) is None
    hindi = VideoCandidate(video_id="v", default_audio_language="hi-IN")
    assert language_match(hindi, None, "hi") == 1.0 and language_match(hindi, None, "en") == 0.0
    assert language_match(VideoCandidate(video_id="v"), None, "en") is None  # unknown language
    ok = TranscriptRecord(video_id="v", status=TranscriptStatus.OK, language_code="en")
    none = TranscriptRecord(video_id="v", status=TranscriptStatus.NO_TRANSCRIPT)
    blocked = TranscriptRecord(video_id="v", status=TranscriptStatus.BLOCKED)
    no_manual = VideoCandidate(video_id="v", caption_flag=False)
    assert caption_match(no_manual, ok) == 1.0 and caption_match(no_manual, none) == 0.0
    assert caption_match(no_manual, None) is None  # caption=false only means "no manual captions"
    assert caption_match(no_manual, blocked) is None  # a block is not evidence of no captions
    assert estimate_video_level("Photosynthesis explained simply") == 0.0
    assert estimate_video_level("Photosynthesis derivation lecture") == 1.0
    assert estimate_video_level("Photosynthesis") is None
    assert difficulty_match(0.0, 1.0) == 0.0 and difficulty_match(None, 0.5) is None


def test_popularity_and_history_are_none_when_unobserved():
    scores = popularity_scores([VideoCandidate(video_id="a", view_count=10), VideoCandidate(video_id="b", view_count=10_000),
                                VideoCandidate(video_id="c")])
    assert scores["a"] == 0.0 and scores["b"] == 1.0 and scores["c"] is None
    assert historical_outcome(None) is None and historical_outcome({"accepted": 0, "dismissed": 0, "ratings": []}) is None
    assert historical_outcome({"accepted": 3, "dismissed": 1, "ratings": []}) == round(4 / 6, 4)
    assert historical_outcome({"accepted": 0, "dismissed": 0, "ratings": [5, 3]}) == 0.8


def test_hard_filters():
    cands = [VideoCandidate(video_id="ok", duration_s=600, privacy_status="public"),
             VideoCandidate(video_id="ok"),  # duplicate
             VideoCandidate(video_id="priv", privacy_status="unlisted"),
             VideoCandidate(video_id="live", live_broadcast_content="upcoming"),
             VideoCandidate(video_id="short", duration_s=20),
             VideoCandidate(video_id="geo", region_blocked=["IN"]),
             VideoCandidate(video_id="geo2", region_allowed=["US"]),
             VideoCandidate(video_id="noembed", embeddable=False),
             VideoCandidate(video_id="done"),
             VideoCandidate(video_id="hindi", default_audio_language="hi"),
             VideoCandidate(video_id="unknown_lang")]
    kept, filtered = hard_filter(cands, region_code="IN", min_duration_s=60, require_embeddable=True,
                                 require_language="en", excluded={"done": "completed"})
    assert [c.video_id for c in kept] == ["ok", "unknown_lang"]  # unknown values never filter
    reasons = {f.video_id: f.reason for f in filtered}
    assert reasons["done"] == "already completed by this learner" and "duplicate" in [f.reason for f in filtered]
    assert {"priv", "live", "short", "geo", "geo2", "noembed", "hindi"} <= set(reasons)


def test_query_includes_gap_concepts_and_level_hint():
    assert build_query(request(objective="photosynthesis", concepts=["stomata", "glucose", "xylem"])) == \
        "photosynthesis stomata glucose explained simply"


# ---- end-to-end pipeline (known correct ranking) ----------------------------------------------------------

def test_transcript_strategy_ranks_the_teaching_video_first_with_grounded_evidence():
    transcripts = FakeTranscripts()
    result = make_pipeline(transcripts=transcripts).run(request(), STRATEGIES["R2"])
    assert result.status == "ok"
    assert [r.video_id for r in result.recommendations] == ["goodTeach01", "popularNoTr"]
    top, second = result.recommendations
    assert top.evidence_level == "transcript" and top.caption_type == "manual" and top.caption_language == "en"
    ev = top.transcript_evidence
    assert ev.excerpts and all(e.start >= 0 for e in ev.excerpts)
    assert set(ev.concepts_covered) == {"chlorophyll", "glucose"} and ev.example_timestamps and ev.step_timestamps
    assert "Transcript (en, manual captions)" in top.explanation and "chlorophyll at" in top.explanation
    # very popular but no transcript: kept, ranked on metadata, and labelled as such
    assert second.evidence_level == "metadata" and second.transcript_status == TranscriptStatus.TRANSCRIPTS_DISABLED
    assert second.score_breakdown["transcript_relevance"] is None and "title and description only" in second.explanation
    reasons = {f.video_id: f.reason for f in result.filtered}
    assert reasons["songVideo01"].startswith("transcript relevance")  # topic in title, but transcript off-topic
    assert reasons["cookPasta01"].startswith("topic similarity")
    assert reasons["privateVid1"] == "privacy status private" and reasons["shortClip01"] == "shorter than 60s"
    # two-stage: transcripts only for the budget, in pre-rank order, in the learner's language only (the language
    # the chosen video is later indexed in, so a transcript the session cannot use never counts as evidence)
    assert len(transcripts.calls) == STRATEGIES["R2"].transcript_budget and transcripts.calls[0][1] == ("en",)


def test_metadata_only_strategy_fetches_no_transcripts():
    transcripts = FakeTranscripts()
    result = make_pipeline(transcripts=transcripts).run(request(), STRATEGIES["R1"])
    assert transcripts.calls == [] and result.metrics.transcripts_attempted == 0
    assert {r.evidence_level for r in result.recommendations} == {"metadata"}
    assert all(r.transcript_status == TranscriptStatus.NOT_ATTEMPTED for r in result.recommendations)


def test_transcripts_disabled_by_config_keeps_pipeline_working():
    result = make_pipeline(transcripts_enabled=False).run(request(), STRATEGIES["R2"])
    assert result.recommendations and all(r.transcript_status == TranscriptStatus.DISABLED_BY_CONFIG
                                          for r in result.recommendations)


def test_excluded_videos_and_count():
    result = make_pipeline().run(request(excluded={"goodTeach01": "dismissed"}, count=1), STRATEGIES["R2"])
    assert [r.video_id for r in result.recommendations] == ["popularNoTr"]
    assert any(f.video_id == "goodTeach01" and "dismissed" in f.reason for f in result.filtered)


def test_history_lookup_feeds_historical_outcome():
    seen = []

    def lookup(ids):
        seen.extend(ids)
        return {"popularNoTr": {"accepted": 5, "dismissed": 0, "ratings": [5]}}

    result = make_pipeline().run(request(), STRATEGIES["R2"], lookup)
    popular = next(r for r in result.recommendations if r.video_id == "popularNoTr")
    assert popular.score_breakdown["historical_outcome"] == pytest.approx((6 / 7 + 1) / 2, abs=1e-4)
    assert "cookPasta01" in seen  # looked up after hard filters, before ranking


def test_search_failure_returns_failed_result_with_metrics():
    api = FakeYouTubeApi()
    api.fail["search"] = api_error(403, "quotaExceeded")
    result = make_pipeline(api).run(request(), STRATEGIES["R2"])
    assert result.status == "failed" and result.recommendations == [] and result.extra["error_kind"] == "quota_exceeded"
    assert result.metrics.api_failures == 1 and result.metrics.rates()["api_failure_rate"] == 1.0
    assert "search failed" in result.errors[0]


def test_metadata_failure_degrades_to_search_snippets():
    api = FakeYouTubeApi()
    api.fail["videos"] = api_error(500, "backendError")
    result = make_pipeline(api).run(request(), STRATEGIES["R2"])
    assert result.status == "degraded" and result.recommendations
    assert any("metadata unavailable" in e for e in result.errors)


class BrokenEmbedder(HashingEmbedder):
    def embed_query(self, text):
        raise EmbeddingError("ollama down")


def test_embedder_failure_falls_back_to_keyword_overlap():
    result = make_pipeline(embedder=BrokenEmbedder()).run(request(), STRATEGIES["R2"])
    assert result.status == "degraded" and result.recommendations
    assert all(r.score_breakdown["transcript_relevance"] is None for r in result.recommendations)
    assert any("keyword overlap" in e for e in result.errors)


def test_empty_search_results():
    result = make_pipeline(FakeYouTubeApi(order=[])).run(request(), STRATEGIES["R2"])
    assert result.status == "empty" and result.recommendations == []


def test_pipeline_instrumentation():
    result = make_pipeline().run(request(), STRATEGIES["R2"])
    m = result.metrics
    assert m.api_calls == 2 and m.quota_units == 101 and m.candidates_found == 6 and m.candidates_filtered == 4
    assert m.transcripts_attempted == 4 and m.transcripts_ok == 2 and m.embeddings > 0
    t = m.timings
    assert None not in (t.search_ms, t.metadata_ms, t.transcript_ms, t.embedding_ms, t.ranking_ms, t.total_ms)
    assert m.rates()["transcript_success_rate"] == 0.5 and m.rates()["candidate_filtering_rate"] == round(4 / 6, 4)


def test_slow_transcripts_are_dropped_after_the_time_budget():
    import time

    class SlowTranscripts(FakeTranscripts):
        def fetch(self, video_id, languages):
            if video_id == "goodTeach01":
                time.sleep(1.0)
            return super().fetch(video_id, languages)

    started = time.perf_counter()
    result = make_pipeline(transcripts=SlowTranscripts(), transcript_time_budget_s=0.3).run(request(), STRATEGIES["R2"])
    assert time.perf_counter() - started < 0.9  # did not wait for the slow fetch
    good = next(r for r in result.recommendations if r.video_id == "goodTeach01")
    assert good.transcript_status == TranscriptStatus.FAILED and good.evidence_level == "metadata"
    assert "no transcript evidence (failed)" in good.explanation


def test_without_a_learner_language_transcripts_use_the_configured_languages():
    transcripts = FakeTranscripts()
    make_pipeline(transcripts=transcripts).run(request(language_code=None), STRATEGIES["R2"])
    assert transcripts.calls and all(langs == ("en", "hi") for _, langs in transcripts.calls)


def test_required_transcript_drops_videos_without_a_usable_track_even_for_metadata_only():
    transcripts = FakeTranscripts()
    result = make_pipeline(transcripts=transcripts, require_transcript=True).run(request(), STRATEGIES["R1"])
    assert result.recommendations and all(r.video_id in transcripts.records for r in result.recommendations)
    reasons = {f.video_id: f.reason for f in result.filtered}
    assert reasons["popularNoTr"] == "no transcript track in en (transcripts disabled)"
    assert transcripts.calls == [] and transcripts.checks  # R1 lists tracks but downloads no transcript
    assert all(langs == ("en",) for _, langs in transcripts.checks)


def test_a_track_check_that_cannot_tell_keeps_the_video():
    class Unsure(FakeTranscripts):
        def available(self, video_id, languages):
            raise TimeoutError("listing took too long")

    plain = make_pipeline(transcripts=FakeTranscripts()).run(request(), STRATEGIES["R1"])
    unsure = make_pipeline(transcripts=Unsure(), require_transcript=True).run(request(), STRATEGIES["R1"])
    assert [r.video_id for r in unsure.recommendations] == [r.video_id for r in plain.recommendations]
