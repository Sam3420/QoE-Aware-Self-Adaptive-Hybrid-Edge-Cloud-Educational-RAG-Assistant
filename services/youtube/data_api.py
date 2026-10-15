"""YouTube Data API v3 client: search.list for candidates, videos.list for metadata.

- `part` and `fields` limit every response to what the ranking uses.
- Quota: search.list costs 100 units per page and videos.list 1 unit per call (≤50 ids). A local
  estimate per Pacific day stops calls before the configured daily quota is exhausted.
- API errors are mapped to YouTubeApiError.kind: quota_exceeded, invalid_credentials, forbidden,
  bad_request, not_configured, timeout, network, http_error, invalid_response.
"""
import json
import logging
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from services.youtube.cache import JsonFileCache
from telemetry import tracing
from services.youtube.models import VideoCandidate

logger = logging.getLogger(__name__)

SEARCH_COST = 100
VIDEOS_COST = 1
MAX_IDS_PER_CALL = 50
SEARCH_FIELDS = ("nextPageToken,items(id/videoId,snippet(publishedAt,channelId,title,description,"
                 "thumbnails/medium/url,thumbnails/high/url,channelTitle,liveBroadcastContent))")
VIDEOS_PARTS = "snippet,contentDetails,statistics,status"
VIDEOS_FIELDS = ("items(id,snippet(publishedAt,channelId,title,description,thumbnails/medium/url,"
                 "thumbnails/high/url,channelTitle,tags,categoryId,liveBroadcastContent,defaultLanguage,"
                 "defaultAudioLanguage),contentDetails(duration,caption,definition,regionRestriction),"
                 "statistics(viewCount,likeCount,commentCount),status(uploadStatus,privacyStatus,embeddable,"
                 "madeForKids))")
VIDEO_DURATIONS = {"any", "short", "medium", "long"}  # short < 4 min, medium 4–20 min, long > 20 min
VIDEO_CAPTIONS = {"any", "closedCaption", "none"}
_ISO_DURATION = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PACIFIC = timezone(timedelta(hours=-8))  # quota resets at midnight Pacific; DST ignored (estimate)


class YouTubeApiError(RuntimeError):
    """A failed Data API call, classified by `kind`."""

    def __init__(self, kind: str, message: str, status_code: int | None = None) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.status_code = status_code


def parse_iso8601_duration(value: str | None) -> int | None:
    """Convert contentDetails.duration (e.g. 'PT1H2M3S') to seconds.

    None if absent or malformed, and for 'P0D', which YouTube returns for live/upcoming broadcasts
    whose duration is not known yet.
    """
    if not value:
        return None
    match = _ISO_DURATION.match(value)
    if not match or not any(match.groups()):
        return None
    days, hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    return (days * 86400 + hours * 3600 + minutes * 60 + seconds) or None


def _int_or_none(value: Any) -> int | None:
    """Statistics are returned as strings and may be missing."""
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _parse_dt(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None
    except ValueError:
        return None


def _thumbnail(snippet: dict[str, Any]) -> str | None:
    thumbs = snippet.get("thumbnails") or {}
    for size in ("high", "medium", "default"):
        url = (thumbs.get(size) or {}).get("url")
        if url:
            return url
    return None


class QuotaTracker:
    """Local estimate of units spent today (Pacific time), persisted to a JSON file."""

    def __init__(self, path: Path | None, daily_units: int, reserve_units: int) -> None:
        self.path = path
        self.daily_units = daily_units
        self.reserve_units = reserve_units
        self._state = {"day": self._today(), "units": 0}
        if path and path.exists():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved.get("day") == self._state["day"]:
                    self._state = saved
            except (OSError, ValueError):
                pass

    @staticmethod
    def _today() -> str:
        return datetime.now(_PACIFIC).date().isoformat()

    @property
    def used(self) -> int:
        if self._state["day"] != self._today():
            self._state = {"day": self._today(), "units": 0}
        return int(self._state["units"])

    def can_spend(self, units: int) -> bool:
        """True if spending `units` keeps the estimate below the daily quota minus the reserve."""
        return self.used + units <= self.daily_units - self.reserve_units

    def spend(self, units: int) -> None:
        self._state = {"day": self._today(), "units": self.used + units}
        if self.path:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self._state), encoding="utf-8")
            except OSError as exc:
                logger.warning("quota tracker not saved: %s", exc)


@dataclass
class CallStats:
    """Per-run accounting of API calls (reset by the pipeline for each run)."""

    calls: int = 0
    failures: int = 0
    units: int = 0
    bytes: int = 0
    search_ms: float = 0.0
    metadata_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class SearchPage:
    """One page of search.list results."""

    candidates: list[VideoCandidate]
    next_page_token: str | None


class YouTubeDataClient:
    """Thin, validated wrapper over search.list and videos.list."""

    def __init__(self, *, api_key: str | None, base_url: str, timeout_s: float, cache: JsonFileCache | None,
                 quota: QuotaTracker, http: httpx.Client | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.quota = quota
        self.http = http or httpx.Client(timeout=timeout_s)
        self.stats = CallStats()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def reset_stats(self) -> CallStats:
        """Start a new accounting period and return the previous one."""
        previous, self.stats = self.stats, CallStats()
        return previous

    # ---- low level -------------------------------------------------------------------------------

    def _get(self, resource: str, params: dict[str, Any], cost: int) -> dict[str, Any]:
        """GET one resource with quota accounting and error classification."""
        if not self.api_key:
            raise YouTubeApiError("not_configured", "YOUTUBE_API_KEY is not set")
        if not self.quota.can_spend(cost):
            raise YouTubeApiError("quota_exceeded", f"local daily quota estimate reached ({self.quota.used} units "
                                  f"used of {self.quota.daily_units}, reserve {self.quota.reserve_units})")
        self.stats.calls += 1
        self.quota.spend(cost)  # YouTube charges failed requests too
        self.stats.units += cost
        try:
            resp = self.http.get(f"{self.base_url}/{resource}", params={**params, "key": self.api_key})
        except httpx.TimeoutException as exc:
            self.stats.failures += 1
            raise YouTubeApiError("timeout", f"{resource} timed out") from exc
        except httpx.HTTPError as exc:
            self.stats.failures += 1
            raise YouTubeApiError("network", f"{resource}: {type(exc).__name__}") from exc
        self.stats.bytes += len(resp.content)
        if resp.status_code != 200:
            self.stats.failures += 1
            raise self._classify(resource, resp)
        try:
            data = resp.json()
        except ValueError as exc:
            self.stats.failures += 1
            raise YouTubeApiError("invalid_response", f"{resource} returned non-JSON") from exc
        if not isinstance(data, dict) or not isinstance(data.get("items", []), list):
            self.stats.failures += 1
            raise YouTubeApiError("invalid_response", f"{resource} returned an unexpected shape")
        return data

    @staticmethod
    def _classify(resource: str, resp: httpx.Response) -> YouTubeApiError:
        """Map an error response to a YouTubeApiError (never echoes the API key)."""
        try:
            err = resp.json().get("error", {})
            reasons = {e.get("reason") for e in err.get("errors", []) if isinstance(e, dict)}
            message = str(err.get("message", ""))[:200]
        except ValueError:
            reasons, message = set(), resp.text[:200]
        if reasons & {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded", "userRateLimitExceeded"}:
            kind = "quota_exceeded"
        elif reasons & {"keyInvalid", "keyExpired"} or resp.status_code == 401:
            kind = "invalid_credentials"
        elif resp.status_code == 403:
            kind = "forbidden"  # e.g. accessNotConfigured: the API is not enabled for the key's project
        elif resp.status_code == 400:
            kind = "bad_request"
        else:
            kind = "http_error"
        return YouTubeApiError(kind, f"{resource} HTTP {resp.status_code} {sorted(r for r in reasons if r)} {message}",
                               resp.status_code)

    # ---- search.list -----------------------------------------------------------------------------

    def search_page(self, query: str, *, max_results: int = 25, page_token: str | None = None,
                    relevance_language: str | None = None, region_code: str | None = None,
                    video_duration: str = "any", video_caption: str = "any",
                    safe_search: str = "strict") -> SearchPage:
        """One search.list page of videos (type=video, so the duration/caption filters are allowed)."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if video_duration not in VIDEO_DURATIONS or video_caption not in VIDEO_CAPTIONS:
            raise ValueError("unsupported videoDuration/videoCaption value")
        params: dict[str, Any] = {
            "part": "snippet", "type": "video", "q": query.strip()[:200],
            "maxResults": max(1, min(int(max_results), 50)), "safeSearch": safe_search,
            "fields": SEARCH_FIELDS,
        }
        if page_token:
            params["pageToken"] = page_token
        if relevance_language:
            params["relevanceLanguage"] = relevance_language
        if region_code:
            params["regionCode"] = region_code
        if video_duration != "any":
            params["videoDuration"] = video_duration
        if video_caption != "any":
            params["videoCaption"] = video_caption

        cache_key = json.dumps(params, sort_keys=True)
        cached = self.cache.get("search", cache_key) if self.cache else None
        started = time.perf_counter()
        if cached is None:
            data = self._get("search", params, SEARCH_COST)
            if self.cache:
                self.cache.set("search", cache_key, data)
        else:
            data = cached
        self.stats.search_ms += (time.perf_counter() - started) * 1000

        candidates = []
        for item in data.get("items", []):
            video_id = (item.get("id") or {}).get("videoId") if isinstance(item, dict) else None
            if not video_id or not _VIDEO_ID.match(video_id):
                continue  # channels/playlists or malformed ids
            snippet = item.get("snippet") or {}
            candidates.append(VideoCandidate(
                video_id=video_id, title=str(snippet.get("title", "")), description=str(snippet.get("description", "")),
                channel_id=snippet.get("channelId"), channel_title=snippet.get("channelTitle"),
                published_at=_parse_dt(snippet.get("publishedAt")), thumbnail_url=_thumbnail(snippet),
                live_broadcast_content=snippet.get("liveBroadcastContent"),
            ))
        return SearchPage(candidates, data.get("nextPageToken"))

    def search(self, query: str, *, max_results: int, max_pages: int = 1, **kwargs: Any) -> list[VideoCandidate]:
        """Up to max_results unique videos across at most max_pages pages, in search order."""
        units_before = self.stats.units
        with tracing.observation("search-youtube", as_type="tool",
                                 input={"query": query, "max_results": max_results, "max_pages": max_pages, **kwargs},
                                 metadata={"api": "search.list"}) as obs:
            found: dict[str, VideoCandidate] = {}
            token: str | None = None
            for _ in range(max_pages):
                page = self.search_page(query, max_results=min(50, max_results - len(found)), page_token=token,
                                        **kwargs)
                for cand in page.candidates:
                    if cand.video_id not in found:  # duplicate removal across pages
                        found[cand.video_id] = cand.model_copy(update={"search_rank": len(found)})
                token = page.next_page_token
                if not token or len(found) >= max_results:
                    break
            result = list(found.values())[:max_results]
            obs.update(output=[{"video_id": c.video_id, "title": c.title} for c in result],
                       metadata={"api": "search.list", "quota_units": self.stats.units - units_before})
            return result

    # ---- videos.list -----------------------------------------------------------------------------

    def get_videos(self, video_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        """videos.list for up to 50 ids per call; returns raw items by id (missing ids are not returned)."""
        ids = list(dict.fromkeys(v for v in video_ids if _VIDEO_ID.match(v)))
        with tracing.observation("fetch-video-metadata", as_type="tool", input={"video_ids": ids},
                                 metadata={"api": "videos.list", "parts": VIDEOS_PARTS}) as obs:
            items = self._get_videos(ids)
            obs.update(output={"returned": len(items), "missing": [v for v in ids if v not in items]})
            return items

    def _get_videos(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        items: dict[str, dict[str, Any]] = {}
        to_fetch = []
        for vid in ids:
            cached = self.cache.get("videos", vid) if self.cache else None
            if cached is not None:
                items[vid] = cached
            else:
                to_fetch.append(vid)
        started = time.perf_counter()
        for i in range(0, len(to_fetch), MAX_IDS_PER_CALL):
            batch = to_fetch[i:i + MAX_IDS_PER_CALL]
            data = self._get("videos", {"part": VIDEOS_PARTS, "id": ",".join(batch), "fields": VIDEOS_FIELDS,
                                        "maxResults": len(batch)}, VIDEOS_COST)
            for item in data.get("items", []):
                if isinstance(item, dict) and item.get("id") in batch:
                    items[item["id"]] = item
                    if self.cache:
                        self.cache.set("videos", item["id"], item)
        self.stats.metadata_ms += (time.perf_counter() - started) * 1000
        return items

    @staticmethod
    def merge_details(candidate: VideoCandidate, item: dict[str, Any] | None) -> VideoCandidate:
        """Return the candidate updated with videos.list fields; fields YouTube omitted stay None."""
        if not item:
            return candidate
        snippet = item.get("snippet") or {}
        details = item.get("contentDetails") or {}
        stats = item.get("statistics") or {}
        status = item.get("status") or {}
        region = details.get("regionRestriction") or {}
        caption = details.get("caption")
        update: dict[str, Any] = {
            "title": snippet.get("title") or candidate.title,
            "description": snippet.get("description") or candidate.description,  # search truncates descriptions
            "channel_id": snippet.get("channelId") or candidate.channel_id,
            "channel_title": snippet.get("channelTitle") or candidate.channel_title,
            "published_at": _parse_dt(snippet.get("publishedAt")) or candidate.published_at,
            "thumbnail_url": _thumbnail(snippet) or candidate.thumbnail_url,
            "category_id": snippet.get("categoryId"),
            "tags": [str(t) for t in snippet.get("tags", []) if isinstance(t, str)],
            "live_broadcast_content": snippet.get("liveBroadcastContent") or candidate.live_broadcast_content,
            "default_language": snippet.get("defaultLanguage"),
            "default_audio_language": snippet.get("defaultAudioLanguage"),
            "duration_s": parse_iso8601_duration(details.get("duration")),
            "caption_flag": {"true": True, "false": False}.get(str(caption).lower()) if caption is not None else None,
            "definition": details.get("definition"),
            "region_allowed": region.get("allowed"),
            "region_blocked": region.get("blocked"),
            "view_count": _int_or_none(stats.get("viewCount")),
            "like_count": _int_or_none(stats.get("likeCount")),
            "comment_count": _int_or_none(stats.get("commentCount")),
            "privacy_status": status.get("privacyStatus"),
            "upload_status": status.get("uploadStatus"),
            "embeddable": status.get("embeddable") if isinstance(status.get("embeddable"), bool) else None,
            "made_for_kids": status.get("madeForKids") if isinstance(status.get("madeForKids"), bool) else None,
            "has_details": True,
        }
        return candidate.model_copy(update=update)
