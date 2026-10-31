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


def _translation(listing: Any, languages: Sequence[str]) -> tuple[Any, Any]:
    """YouTube's own machine translation into the first preferred language, from a translatable track (manual tracks
    first); (None, None) when no track can be translated into it."""
    target = languages[0] if languages else None
    for source in sorted(listing, key=lambda t: t.is_generated):
        codes = {getattr(lang, "language_code", None) for lang in getattr(source, "translation_languages", None) or []}
        if target and getattr(source, "is_translatable", False) and target in codes:
            return source.translate(target), source
    return None, None


class YouTubeTranscriptApiAdapter:
    """Fetches the preferred transcript track: manual in a preferred language, else auto-generated, else (with
    translate_fallback) YouTube's translation of another track into the first preferred language."""

    def __init__(self, api: Any | None = None, *, translate_fallback: bool = False) -> None:
        if api is None:
            from youtube_transcript_api import YouTubeTranscriptApi  # optional dependency

            api = YouTubeTranscriptApi()
        self.api = api
        self.translate_fallback = translate_fallback
        self._blocked: tuple[str, float] | None = None  # (error name, monotonic time) of YouTube's last block

    # After a block, no requests for BLOCK_COOLDOWN_S (more requests prolong it), then the next one tries again:
    # a latch for the whole process kept every video "blocked" long after YouTube had lifted the block.
    BLOCK_COOLDOWN_S = 900.0

    @property
    def blocked(self) -> str | None:
        """The error name while YouTube's block is assumed to last, else None."""
        if self._blocked and time.monotonic() - self._blocked[1] < self.BLOCK_COOLDOWN_S:
            return self._blocked[0]
        return None

    def _block(self, exc: Exception) -> None:
        self._blocked = (type(exc).__name__, time.monotonic())

    def _pick(self, listing: Any, languages: Sequence[str]) -> tuple[Any, Any]:
        """(track to fetch, the track it is translated from or None); (None, None) when nothing is usable."""
        for finder in (listing.find_manually_created_transcript, listing.find_generated_transcript):
            try:
                return finder(list(languages)), None
            except Exception as exc:  # NoTranscriptFound for this kind of track: try the next kind
                if _status_for(exc) != TranscriptStatus.NO_TRANSCRIPT:
                    raise
        return _translation(listing, languages) if self.translate_fallback else (None, None)

    def available(self, video_id: str, languages: Sequence[str]) -> TranscriptStatus:
        """Whether fetch() would find a usable track, by listing the tracks only (nothing is downloaded): OK,
        NO_TRANSCRIPT, or the status of the error (BLOCKED, TRANSCRIPTS_DISABLED, ...)."""
        if self.blocked:
            return TranscriptStatus.BLOCKED
        try:
            track, _ = self._pick(self.api.list(video_id), languages)
            return TranscriptStatus.OK if track is not None else TranscriptStatus.NO_TRANSCRIPT
        except Exception as exc:
            status = _status_for(exc)
            if status == TranscriptStatus.BLOCKED:
                self._block(exc)
            return status

    def fetch(self, video_id: str, languages: Sequence[str]) -> TranscriptRecord:
        """Return the normalised transcript, or a record explaining why there is none."""
        if self.blocked:
            return TranscriptRecord(video_id=video_id, status=TranscriptStatus.BLOCKED,
                                    error=f"skipped: YouTube is blocking transcript requests from this network ({self.blocked}); "
                                          f"retried automatically after a few minutes")
        try:
            listing = self.api.list(video_id)
            track, source = self._pick(listing, languages)
            if track is None:
                available = ", ".join(sorted({f"{t.language_code}{'*' if t.is_generated else ''}" for t in listing}))
                return TranscriptRecord(video_id=video_id, status=TranscriptStatus.NO_TRANSCRIPT, retrieved_at=_now(),
                                        error=f"no track in {list(languages)}; available: {available or 'none'}")
            fetched = track.fetch()
            segments = [TranscriptSegment(start=max(0.0, float(s.start)), duration=max(0.0, float(s.duration)),
                                          text=str(s.text)) for s in fetched.snippets]
            kind = "auto" if fetched.is_generated else "manual"
            if source is not None:  # machine-translated by YouTube: say so wherever the caption type is shown
                kind = f"{'auto' if source.is_generated else 'manual'}-translated from {source.language_code}"
            return TranscriptRecord(
                video_id=video_id, status=TranscriptStatus.OK if segments else TranscriptStatus.NO_TRANSCRIPT,
                language_code=fetched.language_code, language=fetched.language,
                caption_type=kind, segments=segments, retrieved_at=_now(),
                error=None if segments else "transcript track is empty",
            )
        except Exception as exc:
            status = _status_for(exc)
            first_line = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else type(exc).__name__
            if status == TranscriptStatus.BLOCKED:
                self._block(exc)
                logger.warning("transcript requests blocked by YouTube; pausing them for %.0f min: %s",
                               self.BLOCK_COOLDOWN_S / 60, first_line)
            return TranscriptRecord(video_id=video_id, status=status, retrieved_at=_now(),
                                    error=f"{type(exc).__name__}: {first_line}")


# definite "no usable track" answers, safe to remember (blocks, timeouts and other failures are not)
KNOWN_TRACK_VERDICTS = frozenset({TranscriptStatus.NO_TRANSCRIPT, TranscriptStatus.TRANSCRIPTS_DISABLED})


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
        known = self.cache.get("captions", key) if self.cache else None
        if known in {s.value for s in KNOWN_TRACK_VERDICTS}:  # listed earlier: no usable track, nothing to download
            return TranscriptRecord(video_id=video_id, status=TranscriptStatus(known), retrieved_at=_now(),
                                    from_cache=True, error=f"no usable track (cached check: {known})")
        record = self.inner.fetch(video_id, languages)
        if record.status == TranscriptStatus.OK and self.cache:
            self.cache.set("transcripts", key, record.model_dump(mode="json"))
        elif record.status in KNOWN_TRACK_VERDICTS and self.cache:  # remembered for available()
            self.cache.set("captions", key, record.status.value)
        return record

    def available(self, video_id: str, languages: Sequence[str]) -> TranscriptStatus:
        """OK for a cached transcript; else a cached track verdict (namespace "captions"); else the inner
        fetcher's track listing, whose definite verdicts are cached (NOT_ATTEMPTED if it cannot list tracks)."""
        key = f"{video_id}|{','.join(languages)}"
        if self.cache:
            if self.cache.get("transcripts", key) is not None:
                return TranscriptStatus.OK
            known = self.cache.get("captions", key)
            if known in {s.value for s in KNOWN_TRACK_VERDICTS} | {TranscriptStatus.OK.value}:
                return TranscriptStatus(known)
        check = getattr(self.inner, "available", None)
        status = check(video_id, languages) if check else TranscriptStatus.NOT_ATTEMPTED
        if self.cache and (status in KNOWN_TRACK_VERDICTS or status == TranscriptStatus.OK):
            self.cache.set("captions", key, status.value)
        return status


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
