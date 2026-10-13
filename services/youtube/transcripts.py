"""Transcript retrieval behind a replaceable adapter.

YouTubeTranscriptApiAdapter wraps the unofficial `youtube-transcript-api` (v1.x: instance API,
`list()` → `TranscriptList` → `Transcript.fetch()`). It relies on YouTube internals that can change,
so every failure becomes a TranscriptStatus and the video stays in the ranking on metadata alone.
Access restrictions (age gates, logins, IP blocks) are reported, never bypassed: after a block,
retrieval stops for the rest of the run instead of retrying.
"""
import logging
import re
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Protocol

from services.youtube.cache import JsonFileCache
from services.youtube.models import TranscriptRecord, TranscriptSegment, TranscriptStatus

logger = logging.getLogger(__name__)


class TranscriptFetcher(Protocol):
    """What the pipeline needs from a transcript source."""

    def fetch(self, video_id: str, languages: Sequence[str]) -> TranscriptRecord: ...


class DisabledTranscriptFetcher:
    """Used when TRANSCRIPTS_ENABLED=false or the library is not installed."""

    def __init__(self, reason: str = "transcript retrieval disabled by configuration") -> None:
        self.reason = reason

    def fetch(self, video_id: str, languages: Sequence[str]) -> TranscriptRecord:
        return TranscriptRecord(video_id=video_id, status=TranscriptStatus.DISABLED_BY_CONFIG, error=self.reason)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _status_for(exc: Exception) -> TranscriptStatus:
    """Map youtube-transcript-api exceptions (by class name, so a library upgrade cannot break imports)."""
    names = {cls.__name__ for cls in type(exc).__mro__}
    if names & {"RequestBlocked", "IpBlocked"}:
        return TranscriptStatus.BLOCKED
    if "TranscriptsDisabled" in names:
        return TranscriptStatus.TRANSCRIPTS_DISABLED
    if "NoTranscriptFound" in names:
        return TranscriptStatus.NO_TRANSCRIPT
    if names & {"VideoUnavailable", "VideoUnplayable", "InvalidVideoId"}:
        return TranscriptStatus.VIDEO_UNAVAILABLE
    if names & {"AgeRestricted", "PoTokenRequired"}:
        return TranscriptStatus.AGE_RESTRICTED
    return TranscriptStatus.FAILED


class YouTubeTranscriptApiAdapter:
    """Fetches the preferred transcript track: manual in a preferred language, else auto-generated."""

    def __init__(self, api: Any | None = None) -> None:
        if api is None:
            from youtube_transcript_api import YouTubeTranscriptApi  # optional dependency

            api = YouTubeTranscriptApi()
        self.api = api
        self.blocked: str | None = None  # set after YouTube blocks requests; no further calls this process

    def fetch(self, video_id: str, languages: Sequence[str]) -> TranscriptRecord:
        """Return the normalised transcript, or a record explaining why there is none."""
        if self.blocked:
            return TranscriptRecord(video_id=video_id, status=TranscriptStatus.BLOCKED,
                                    error=f"skipped: YouTube blocked transcript requests earlier ({self.blocked})")
        try:
            listing = self.api.list(video_id)
            track = None
            for finder in (listing.find_manually_created_transcript, listing.find_generated_transcript):
                try:
                    track = finder(list(languages))
                    break
                except Exception as exc:  # NoTranscriptFound for this kind of track: try the next kind
                    if _status_for(exc) != TranscriptStatus.NO_TRANSCRIPT:
                        raise
            if track is None:
                available = ", ".join(sorted({f"{t.language_code}{'*' if t.is_generated else ''}" for t in listing}))
                return TranscriptRecord(video_id=video_id, status=TranscriptStatus.NO_TRANSCRIPT, retrieved_at=_now(),
                                        error=f"no track in {list(languages)}; available: {available or 'none'}")
            fetched = track.fetch()
            segments = [TranscriptSegment(start=max(0.0, float(s.start)), duration=max(0.0, float(s.duration)),
                                          text=str(s.text)) for s in fetched.snippets]
            return TranscriptRecord(
                video_id=video_id, status=TranscriptStatus.OK if segments else TranscriptStatus.NO_TRANSCRIPT,
                language_code=fetched.language_code, language=fetched.language,
                caption_type="auto" if fetched.is_generated else "manual", segments=segments, retrieved_at=_now(),
                error=None if segments else "transcript track is empty",
            )
        except Exception as exc:
            status = _status_for(exc)
            first_line = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else type(exc).__name__
            if status == TranscriptStatus.BLOCKED:
                self.blocked = type(exc).__name__
                logger.warning("transcript requests blocked by YouTube; disabling retrieval: %s", first_line)
            return TranscriptRecord(video_id=video_id, status=status, retrieved_at=_now(),
                                    error=f"{type(exc).__name__}: {first_line}")


class CachedTranscriptFetcher:
    """Reuses transcripts within TRANSCRIPT_CACHE_TTL_S; only successful records are cached."""

    def __init__(self, inner: TranscriptFetcher, cache: JsonFileCache | None) -> None:
        self.inner = inner
        self.cache = cache

    def fetch(self, video_id: str, languages: Sequence[str]) -> TranscriptRecord:
        key = f"{video_id}|{','.join(languages)}"
        cached = self.cache.get("transcripts", key) if self.cache else None
        if cached is not None:
            try:
                return TranscriptRecord.model_validate(cached).model_copy(update={"from_cache": True})
            except ValueError:
                pass
        record = self.inner.fetch(video_id, languages)
        if record.status == TranscriptStatus.OK and self.cache:
            self.cache.set("transcripts", key, record.model_dump(mode="json"))
        return record


_NOISE = re.compile(r"\[(?:music|applause|laughter|inaudible|silence|foreign)\]|♪+|&nbsp;", re.IGNORECASE)
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove caption noise markers and collapse whitespace; formulas and terms are kept as written."""
    text = _NOISE.sub(" ", text).replace("&amp;", "&").replace("&#39;", "'").replace("&quot;", '"')
    return _SPACES.sub(" ", text).strip()


def truncate_record(record: TranscriptRecord, max_words: int) -> TranscriptRecord:
    """Keep whole segments up to max_words (bounded processing for very long videos)."""
    kept, words = [], 0
    for seg in record.segments:
        n = len(seg.text.split())
        if words + n > max_words and kept:
            return record.model_copy(update={"segments": kept, "truncated": True})
        kept.append(seg)
        words += n
    return record


def timed_fetch(fetcher: TranscriptFetcher, video_id: str, languages: Sequence[str]) -> tuple[TranscriptRecord, float]:
    """Fetch and return (record, elapsed ms)."""
    started = time.perf_counter()
    record = fetcher.fetch(video_id, languages)
    return record, (time.perf_counter() - started) * 1000
