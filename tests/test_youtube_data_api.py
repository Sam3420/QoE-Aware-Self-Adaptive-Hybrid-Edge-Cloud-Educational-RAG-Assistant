"""YouTube Data API client: supported parameters, parsing, pagination, errors, quota and cache (all mocked)."""
import httpx
import pytest

from services.youtube.cache import JsonFileCache
from services.youtube.data_api import YouTubeApiError, YouTubeDataClient, parse_iso8601_duration
from services.youtube.models import VideoCandidate
from tests.youtube_fakes import FakeYouTubeApi, api_error, make_client


def test_search_uses_supported_parameters_and_fields():
    api = FakeYouTubeApi()
    client = make_client(api)
    client.search_page("photosynthesis", max_results=5, relevance_language="en", region_code="IN",
                       video_duration="medium", video_caption="closedCaption", safe_search="strict")
    params = api.requests[0].url.params
    assert api.requests[0].url.path.endswith("/search")
    assert params["part"] == "snippet" and params["type"] == "video" and params["q"] == "photosynthesis"
    assert params["maxResults"] == "5" and params["relevanceLanguage"] == "en" and params["regionCode"] == "IN"
    assert params["videoDuration"] == "medium" and params["videoCaption"] == "closedCaption"  # valid with type=video
    assert params["safeSearch"] == "strict" and "items(id/videoId" in params["fields"]


def test_any_duration_and_caption_are_omitted():
    api = FakeYouTubeApi()
    make_client(api).search_page("x", max_results=3)
    assert "videoDuration" not in api.requests[0].url.params and "videoCaption" not in api.requests[0].url.params


def test_invalid_filter_values_are_rejected_before_calling():
    api = FakeYouTubeApi()
    with pytest.raises(ValueError):
        make_client(api).search_page("x", video_duration="tiny")
    assert api.requests == []


def test_pagination_and_duplicate_removal():
    api = FakeYouTubeApi(order=["goodTeach01", "songVideo01", "goodTeach01", "cookPasta01", "popularNoTr"])
    found = make_client(api).search("x", max_results=10, max_pages=3)
    assert [c.video_id for c in found] == ["goodTeach01", "songVideo01", "cookPasta01", "popularNoTr"]
    assert [c.search_rank for c in found] == [0, 1, 2, 3]


def test_max_pages_bounds_quota_use():
    api = FakeYouTubeApi()
    client = make_client(api)
    found = client.search("x", max_results=3, max_pages=1)
    assert len(found) == 3 and len(api.requests) == 1 and client.stats.units == 100


def test_videos_list_batches_parts_and_parses_fields():
    api = FakeYouTubeApi()
    client = make_client(api)
    items = client.get_videos(["goodTeach01", "popularNoTr", "not-a-valid-id"])
    params = api.requests[0].url.params
    assert params["part"] == "snippet,contentDetails,statistics,status" and "contentDetails(duration" in params["fields"]
    assert set(items) == {"goodTeach01", "popularNoTr"}  # malformed id never sent
    merged = YouTubeDataClient.merge_details(VideoCandidate(video_id="goodTeach01"), items["goodTeach01"])
    assert merged.duration_s == 570 and merged.view_count == 120000 and merged.like_count == 4000
    assert merged.caption_flag is True and merged.privacy_status == "public" and merged.embeddable is True
    assert merged.default_audio_language == "en" and merged.comment_count is None  # not returned → None
    assert merged.has_details and merged.url == "https://www.youtube.com/watch?v=goodTeach01"


def test_videos_list_splits_requests_at_50_ids(monkeypatch):
    api = FakeYouTubeApi()
    client = make_client(api)
    ids = [f"id{i:09d}" for i in range(60)]
    client.get_videos(ids)
    assert [len(r.url.params["id"].split(",")) for r in api.requests] == [50, 10]
    assert client.stats.units == 2


@pytest.mark.parametrize(("value", "seconds"), [("PT9M30S", 570), ("PT1H2M3S", 3723), ("PT45S", 45), ("P1DT1M", 86460),
                                               ("P0D", None), ("", None), (None, None), ("garbage", None)])
def test_parse_iso8601_duration(value, seconds):
    assert parse_iso8601_duration(value) == seconds


@pytest.mark.parametrize(("status", "reason", "kind"), [
    (403, "quotaExceeded", "quota_exceeded"), (403, "rateLimitExceeded", "quota_exceeded"),
    (400, "keyInvalid", "invalid_credentials"), (403, "accessNotConfigured", "forbidden"),
    (400, "invalidParameter", "bad_request"), (500, "backendError", "http_error"),
])
def test_api_errors_are_classified_without_leaking_the_key(status, reason, kind):
    api = FakeYouTubeApi()
    api.fail["search"] = api_error(status, reason)
    client = make_client(api, api_key="SECRET-KEY-123")
    with pytest.raises(YouTubeApiError) as info:
        client.search_page("x")
    assert info.value.kind == kind and "SECRET-KEY-123" not in str(info.value)
    assert client.stats.failures == 1


def test_timeout_network_and_non_json_errors():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    for handler, kind in ((timeout, "timeout"), (refused, "network"),
                          (lambda r: httpx.Response(200, content=b"<html>"), "invalid_response")):
        api = FakeYouTubeApi()
        api.fail["search"] = handler
        with pytest.raises(YouTubeApiError) as info:
            make_client(api).search_page("x")
        assert info.value.kind == kind


def test_missing_key_and_local_quota_estimate_stop_calls():
    api = FakeYouTubeApi()
    with pytest.raises(YouTubeApiError) as info:
        make_client(api, api_key=None).search_page("x")
    assert info.value.kind == "not_configured"
    client = make_client(api, daily_units=150, reserve=0)
    client.search_page("x")  # 100 units
    with pytest.raises(YouTubeApiError) as info:
        client.search_page("y")  # would exceed 150
    assert info.value.kind == "quota_exceeded" and len(api.requests) == 1


def test_cache_avoids_repeat_calls_and_respects_ttl(tmp_path):
    api = FakeYouTubeApi()
    cache = JsonFileCache(tmp_path, {"search": 3600, "videos": 3600})
    client = make_client(api, cache=cache)
    client.search_page("photosynthesis", max_results=3)
    client.search_page("photosynthesis", max_results=3)
    client.get_videos(["goodTeach01"])
    client.get_videos(["goodTeach01"])
    assert len(api.requests) == 2 and client.stats.units == 101
    assert cache.hits == 2

    no_store = JsonFileCache(tmp_path / "off", {"search": 0})
    client = make_client(api, cache=no_store)
    client.search_page("photosynthesis", max_results=3)
    assert not (tmp_path / "off").exists()


def test_expired_cache_entries_are_deleted(tmp_path, monkeypatch):
    import services.youtube.cache as cache_mod

    cache = JsonFileCache(tmp_path, {"videos": 10})
    cache.set("videos", "k", {"v": 1})
    now = cache_mod.time.time()
    monkeypatch.setattr(cache_mod.time, "time", lambda: now + 11)
    assert cache.get("videos", "k") is None
    assert not any(tmp_path.rglob("*.json"))
