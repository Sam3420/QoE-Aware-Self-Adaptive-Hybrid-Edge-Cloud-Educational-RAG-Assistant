"""Deterministic fakes for the YouTube Data API (httpx.MockTransport) and transcripts. No network, no quota."""
import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx

from services.youtube.cache import JsonFileCache
from services.youtube.data_api import QuotaTracker, YouTubeDataClient
from services.youtube.models import TranscriptRecord, TranscriptSegment, TranscriptStatus

OBJECTIVE = "photosynthesis chlorophyll light energy glucose"

# Synthetic catalogue with a known correct ranking (ids are 11 characters, like real video ids).
VIDEOS: dict[str, dict[str, Any]] = {
    "goodTeach01": {"title": "Photosynthesis explained simply", "desc": "How plants use light energy and chlorophyll "
                    "to make glucose. Photosynthesis for school students.", "duration": "PT9M30S", "views": "120000",
                    "likes": "4000", "caption": "true", "lang": "en"},
    "popularNoTr": {"title": "Photosynthesis chlorophyll light energy glucose", "desc": "Photosynthesis full lesson: "
                    "chlorophyll, light energy and glucose.", "duration": "PT12M", "views": "9000000",
                    "likes": "300000", "caption": "false", "lang": "en"},
    "songVideo01": {"title": "Photosynthesis song", "desc": "A fun song about photosynthesis.", "duration": "PT3M10S",
                    "views": "50000", "likes": "900", "caption": "true", "lang": "en"},
    "cookPasta01": {"title": "Cooking pasta at home", "desc": "Boil water, add salt and pasta.", "duration": "PT8M",
                    "views": "800000", "likes": "20000", "caption": "true", "lang": "en"},
    "privateVid1": {"title": "Photosynthesis private", "desc": "photosynthesis", "duration": "PT10M", "views": "10",
                    "likes": "1", "caption": "true", "lang": "en", "privacy": "private"},
    "shortClip01": {"title": "Photosynthesis in 30 seconds", "desc": "photosynthesis glucose", "duration": "PT30S",
                    "views": "1000", "likes": "10", "caption": "false", "lang": "en"},
}
SEARCH_ORDER = ["songVideo01", "cookPasta01", "popularNoTr", "privateVid1", "goodTeach01", "shortClip01"]


def transcript(video_id: str, lines: list[str], *, generated: bool = False, lang: str = "en") -> TranscriptRecord:
    """A successful transcript with one 10-second segment per line."""
    return TranscriptRecord(video_id=video_id, status=TranscriptStatus.OK, language_code=lang, language=lang,
                            caption_type="auto" if generated else "manual",
                            segments=[TranscriptSegment(start=10.0 * i, duration=10.0, text=t) for i, t in enumerate(lines)],
                            retrieved_at=datetime.now(timezone.utc))


TRANSCRIPTS: dict[str, TranscriptRecord] = {
    "goodTeach01": transcript("goodTeach01", [
        "Welcome. Today we learn photosynthesis.",
        "Photosynthesis uses light energy captured by chlorophyll in the leaf.",
        "For example, chlorophyll absorbs light energy and the plant makes glucose.",
        "First, water is split. Next, carbon dioxide and light energy make glucose.",
        "So photosynthesis turns light energy into glucose using chlorophyll.",
    ]),
    "songVideo01": transcript("songVideo01", [
        "la la la sing along", "dance dance dance with me", "clap your hands now", "la la la again"], generated=True),
}
FAILURES = {"popularNoTr": TranscriptStatus.TRANSCRIPTS_DISABLED}


class FakeTranscripts:
    """Scripted transcript source; records which videos were requested."""

    def __init__(self, records: dict[str, TranscriptRecord] | None = None,
                 failures: dict[str, TranscriptStatus] | None = None) -> None:
        self.records = TRANSCRIPTS if records is None else records
        self.failures = FAILURES if failures is None else failures
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.checks: list[tuple[str, tuple[str, ...]]] = []

    def fetch(self, video_id: str, languages) -> TranscriptRecord:
        self.calls.append((video_id, tuple(languages)))
        if video_id in self.records:
            return self.records[video_id]
        status = self.failures.get(video_id, TranscriptStatus.NO_TRANSCRIPT)
        return TranscriptRecord(video_id=video_id, status=status, error=f"fake {status.value}")

    def available(self, video_id: str, languages) -> TranscriptStatus:
        """Track listing only: what fetch() would find (recorded separately from fetches)."""
        self.checks.append((video_id, tuple(languages)))
        if video_id in self.records:
            return TranscriptStatus.OK
        return self.failures.get(video_id, TranscriptStatus.NO_TRANSCRIPT)


def video_item(video_id: str) -> dict[str, Any]:
    v = VIDEOS[video_id]
    return {
        "id": video_id,
        "snippet": {"publishedAt": "2024-01-02T03:04:05Z", "channelId": "UCfake", "title": v["title"],
                    "description": v["desc"], "thumbnails": {"high": {"url": f"https://i.ytimg.com/vi/{video_id}/hq.jpg"}},
                    "channelTitle": "Fake Channel", "tags": ["science"], "categoryId": "27",
                    "liveBroadcastContent": "none", "defaultAudioLanguage": v["lang"]},
        "contentDetails": {"duration": v["duration"], "caption": v["caption"], "definition": "hd"},
        "statistics": {"viewCount": v["views"], "likeCount": v["likes"]},
        "status": {"uploadStatus": "processed", "privacyStatus": v.get("privacy", "public"), "embeddable": True,
                   "madeForKids": False},
    }


class FakeYouTubeApi:
    """MockTransport handler for /search and /videos; `fail` injects errors per resource."""

    def __init__(self, order: list[str] | None = None) -> None:
        self.order = SEARCH_ORDER if order is None else order
        self.requests: list[httpx.Request] = []
        self.fail: dict[str, Callable[[httpx.Request], httpx.Response]] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        resource = request.url.path.rsplit("/", 1)[-1]
        if resource in self.fail:
            return self.fail[resource](request)
        params = request.url.params
        if resource == "search":
            n = int(params["maxResults"])
            start = int(params.get("pageToken", "0"))
            ids = self.order[start:start + n]
            body: dict[str, Any] = {"items": [{"id": {"kind": "youtube#video", "videoId": vid},
                                              "snippet": {"title": VIDEOS[vid]["title"], "description": VIDEOS[vid]["desc"][:60],
                                                          "channelTitle": "Fake Channel", "liveBroadcastContent": "none"}}
                                             for vid in ids]}
            if start + n < len(self.order):
                body["nextPageToken"] = str(start + n)
            return httpx.Response(200, json=body)
        if resource == "videos":
            return httpx.Response(200, json={"items": [video_item(v) for v in params["id"].split(",") if v in VIDEOS]})
        return httpx.Response(404, json={"error": {"code": 404, "message": "not found"}})


def api_error(status: int, reason: str) -> Callable[[httpx.Request], httpx.Response]:
    """Build an error response in the Data API's error format."""
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=json.dumps(
            {"error": {"code": status, "message": reason, "errors": [{"reason": reason}]}}))
    return respond


def make_client(api: FakeYouTubeApi, *, cache: JsonFileCache | None = None, daily_units: int = 10_000,
                reserve: int = 0, api_key: str | None = "test-key") -> YouTubeDataClient:
    """A real YouTubeDataClient over the fake transport."""
    return YouTubeDataClient(api_key=api_key, base_url="https://www.googleapis.com/youtube/v3", timeout_s=5,
                             cache=cache, quota=QuotaTracker(None, daily_units, reserve),
                             http=httpx.Client(transport=httpx.MockTransport(api)))


def pipeline_settings(**overrides: Any):
    """PipelineSettings for tests (hashing-embedder calibration, all transcripts enabled)."""
    from config import settings
    from services.youtube.pipeline import PipelineSettings

    base = dict(region_code="IN", safe_search="strict", max_pages=1, transcript_languages=["en", "hi"],
                transcripts_enabled=True, transcript_max_words=12_000, chunk_words=20, chunk_overlap=5,
                max_chunks_per_video=40, min_duration_s=60, duration_min_s=240, duration_max_s=1200,
                require_embeddable=False, require_language_match=False, prior=0.5,
                default_weights=dict(settings.rec_weights))
    return PipelineSettings(**{**base, **overrides})


def make_pipeline(api: FakeYouTubeApi | None = None, transcripts: Any = None, embedder: Any = None,
                  **overrides: Any):
    """A real RecommendationPipeline over fakes; the hashing embedder is calibrated with floor 0, ceil 0.6."""
    from services.vector_store import HashingEmbedder
    from services.youtube.pipeline import RecommendationPipeline
    from services.youtube.relevance import RelevanceScorer

    scorer = RelevanceScorer(embedder or HashingEmbedder(), floor=0.0, ceil=0.6, top_chunks=3, concept_threshold=0.5)
    return RecommendationPipeline(data_client=make_client(api or FakeYouTubeApi()),
                                  transcripts=transcripts if transcripts is not None else FakeTranscripts(),
                                  scorer=scorer, settings=pipeline_settings(**overrides))
