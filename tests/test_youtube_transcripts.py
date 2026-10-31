"""Transcript adapter (mocked library objects), caching, preprocessing and chunking."""
from types import SimpleNamespace

from services.youtube.cache import JsonFileCache
from services.youtube.models import TranscriptStatus
from services.youtube.relevance import chunk_transcript
from services.youtube.transcripts import (
    CachedTranscriptFetcher,
    YouTubeTranscriptApiAdapter,
    clean_text,
    truncate_record,
)
from tests.youtube_fakes import FakeTranscripts, transcript


# Exception classes named like the library's: the adapter maps errors by class name.
class NoTranscriptFound(Exception): ...
class TranscriptsDisabled(Exception): ...
class AgeRestricted(Exception): ...
class RequestBlocked(Exception): ...
class IpBlocked(RequestBlocked): ...
class VideoUnavailable(Exception): ...


def _track(lang, generated, texts=("hello world",)):
    fetched = SimpleNamespace(language_code=lang, language=lang, is_generated=generated,
                              snippets=[SimpleNamespace(text=t, start=i * 2.0, duration=2.0) for i, t in enumerate(texts)])
    return SimpleNamespace(language_code=lang, is_generated=generated, fetch=lambda: fetched)


class FakeListing(list):
    def __init__(self, tracks):
        super().__init__(tracks)

    def _find(self, langs, generated):
        for lang in langs:
            for t in self:
                if t.language_code == lang and t.is_generated == generated:
                    return t
        raise NoTranscriptFound(langs)

    def find_manually_created_transcript(self, langs):
        return self._find(langs, False)

    def find_generated_transcript(self, langs):
        return self._find(langs, True)


class FakeApi:
    def __init__(self, tracks=None, error=None):
        self.tracks, self.error, self.calls = tracks or [], error, 0

    def list(self, video_id):
        self.calls += 1
        if self.error:
            raise self.error
        return FakeListing(self.tracks)


def test_manual_track_preferred_over_auto_in_language_order():
    api = FakeApi([_track("en", True, ["auto text"]), _track("hi", False, ["manual hindi"]), _track("en", False, ["manual en"])])
    record = YouTubeTranscriptApiAdapter(api).fetch("vid", ["en", "hi"])
    assert record.status == TranscriptStatus.OK and record.caption_type == "manual" and record.language_code == "en"
    assert record.text == "manual en" and record.segments[0].start == 0.0


def test_falls_back_to_auto_generated():
    record = YouTubeTranscriptApiAdapter(FakeApi([_track("hi", True, ["auto"])])).fetch("vid", ["en", "hi"])
    assert record.status == TranscriptStatus.OK and record.caption_type == "auto" and record.language_code == "hi"


def test_no_track_in_requested_languages():
    record = YouTubeTranscriptApiAdapter(FakeApi([_track("fr", False)])).fetch("vid", ["en"])
    assert record.status == TranscriptStatus.NO_TRANSCRIPT and "fr" in record.error and not record.segments


def test_failure_statuses_and_block_circuit_breaker():
    for error, status in ((TranscriptsDisabled("v"), TranscriptStatus.TRANSCRIPTS_DISABLED),
                          (AgeRestricted("v"), TranscriptStatus.AGE_RESTRICTED),
                          (VideoUnavailable("v"), TranscriptStatus.VIDEO_UNAVAILABLE),
                          (RuntimeError("parse"), TranscriptStatus.FAILED)):
        record = YouTubeTranscriptApiAdapter(FakeApi(error=error)).fetch("vid", ["en"])
        assert record.status == status and record.error

    api = FakeApi(error=IpBlocked("v"))
    adapter = YouTubeTranscriptApiAdapter(api)
    assert adapter.fetch("a", ["en"]).status == TranscriptStatus.BLOCKED
    second = adapter.fetch("b", ["en"])
    assert second.status == TranscriptStatus.BLOCKED and "skipped" in second.error and api.calls == 1  # no retry


def test_real_library_exception_maps_correctly():
    from youtube_transcript_api import TranscriptsDisabled as RealDisabled

    record = YouTubeTranscriptApiAdapter(FakeApi(error=RealDisabled("abcdefghijk"))).fetch("abcdefghijk", ["en"])
    assert record.status == TranscriptStatus.TRANSCRIPTS_DISABLED


def test_cached_fetcher_reuses_successes_only(tmp_path):
    inner = FakeTranscripts()
    fetcher = CachedTranscriptFetcher(inner, JsonFileCache(tmp_path, {"transcripts": 3600}))
    first = fetcher.fetch("goodTeach01", ["en"])
    second = fetcher.fetch("goodTeach01", ["en"])
    assert not first.from_cache and second.from_cache and second.text == first.text
    fetcher.fetch("popularNoTr", ["en"])
    fetcher.fetch("popularNoTr", ["en"])
    assert [c[0] for c in inner.calls] == ["goodTeach01", "popularNoTr", "popularNoTr"]  # failures retried


def test_clean_text_and_truncation():
    assert clean_text("[Music]  E = mc^2 &amp; more\n[Applause] text") == "E = mc^2 & more text"
    record = transcript("v", ["one two three", "four five", "six"])
    cut = truncate_record(record, max_words=5)
    assert cut.truncated and [s.text for s in cut.segments] == ["one two three", "four five"]


def test_chunking_keeps_timestamps_overlap_and_bounds():
    record = transcript("v", [f"w{i} " * 10 for i in range(10)])  # 100 words, 10 s per segment
    chunks = chunk_transcript(record, words_per_chunk=30, overlap=10, max_chunks=50)
    assert chunks[0].start == 0.0 and chunks[0].text.split()[:2] == ["w0", "w0"]
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]  # overlap
    assert chunks[1].start == 20.0 and chunks[-1].end == 100.0
    bounded = chunk_transcript(record, words_per_chunk=30, overlap=10, max_chunks=2)
    assert len(bounded) == 2 and bounded[0].start == 0.0 and bounded[-1].end == 100.0  # evenly sampled
    assert chunk_transcript(transcript("e", []), words_per_chunk=30, overlap=10, max_chunks=5) == []


def _translatable(lang, generated, translated_texts=("translated text",)):
    """A track YouTube can machine-translate into English."""
    track = _track(lang, generated, ["original"])
    track.is_translatable = True
    track.translation_languages = [SimpleNamespace(language_code="en", language="English")]
    track.translate = lambda code: _track(code, True, translated_texts)
    return track


def test_translation_fallback_when_no_track_in_the_preferred_language():
    api = FakeApi([_translatable("hi", True, ["plants make food"])])
    record = YouTubeTranscriptApiAdapter(api, translate_fallback=True).fetch("vid", ["en"])
    assert record.status == TranscriptStatus.OK and record.language_code == "en"
    assert record.caption_type == "auto-translated from hi" and record.text == "plants make food"


def test_translation_fallback_is_off_by_default_and_prefers_manual_tracks():
    tracks = [_translatable("hi", True, ["from auto"]), _translatable("mr", False, ["from manual"])]
    assert YouTubeTranscriptApiAdapter(FakeApi(tracks)).fetch("vid", ["en"]).status == TranscriptStatus.NO_TRANSCRIPT
    record = YouTubeTranscriptApiAdapter(FakeApi(tracks), translate_fallback=True).fetch("vid", ["en"])
    assert record.caption_type == "manual-translated from mr" and record.text == "from manual"


def test_a_real_track_in_the_preferred_language_beats_a_translation():
    api = FakeApi([_translatable("hi", False), _track("en", True, ["real english"])])
    record = YouTubeTranscriptApiAdapter(api, translate_fallback=True).fetch("vid", ["en"])
    assert record.caption_type == "auto" and record.text == "real english"


def test_available_lists_tracks_without_downloading():
    downloads = []
    track = _track("en", True)
    track.fetch = lambda: downloads.append(1)
    assert YouTubeTranscriptApiAdapter(FakeApi([track])).available("vid", ["en"]) == TranscriptStatus.OK
    assert YouTubeTranscriptApiAdapter(FakeApi([_track("hi", True)])).available("vid", ["en"]) == TranscriptStatus.NO_TRANSCRIPT
    hindi = _translatable("hi", True)
    assert YouTubeTranscriptApiAdapter(FakeApi([hindi]), translate_fallback=True).available("vid", ["en"]) == TranscriptStatus.OK
    blocked = YouTubeTranscriptApiAdapter(FakeApi(error=IpBlocked("blocked")))
    assert blocked.available("vid", ["en"]) == TranscriptStatus.BLOCKED and blocked.blocked == "IpBlocked"
    assert downloads == []


def test_a_block_pauses_requests_only_for_the_cooldown(monkeypatch):
    import services.youtube.transcripts as transcripts_module

    now = [1000.0]
    monkeypatch.setattr(transcripts_module.time, "monotonic", lambda: now[0])
    api = FakeApi(error=IpBlocked("blocked"))
    adapter = YouTubeTranscriptApiAdapter(api)
    assert adapter.available("a", ["en"]) == TranscriptStatus.BLOCKED
    assert adapter.fetch("b", ["en"]).status == TranscriptStatus.BLOCKED and api.calls == 1  # paused: not sent
    now[0] += adapter.BLOCK_COOLDOWN_S + 1
    api.error, api.tracks = None, [_track("en", False)]
    assert adapter.blocked is None and adapter.available("c", ["en"]) == TranscriptStatus.OK and api.calls == 2


def test_a_cached_no_track_verdict_skips_the_download(tmp_path):
    inner = FakeTranscripts()
    fetcher = CachedTranscriptFetcher(inner, JsonFileCache(tmp_path, {"transcripts": 3600, "captions": 3600}))
    assert fetcher.available("popularNoTr", ["en"]) == TranscriptStatus.TRANSCRIPTS_DISABLED
    record = fetcher.fetch("popularNoTr", ["en"])
    assert record.status == TranscriptStatus.TRANSCRIPTS_DISABLED and record.from_cache and inner.calls == []


def test_track_verdicts_are_cached_but_blocks_are_not(tmp_path):
    cache = JsonFileCache(tmp_path, {"transcripts": 3600, "captions": 3600})
    api = FakeApi([_track("hi", True)])
    fetcher = CachedTranscriptFetcher(YouTubeTranscriptApiAdapter(api), cache)
    assert fetcher.available("vid", ["en"]) == TranscriptStatus.NO_TRANSCRIPT
    assert fetcher.available("vid", ["en"]) == TranscriptStatus.NO_TRANSCRIPT and api.calls == 1  # remembered
    failed = CachedTranscriptFetcher(YouTubeTranscriptApiAdapter(FakeApi([_track("hi", True)])), cache)
    assert failed.fetch("other", ["en"]).status == TranscriptStatus.NO_TRANSCRIPT  # a failed download counts too
    assert cache.get("captions", "other|en") == "no_transcript"
    blocked = CachedTranscriptFetcher(YouTubeTranscriptApiAdapter(FakeApi(error=IpBlocked("x"))), cache)
    assert blocked.available("third", ["en"]) == TranscriptStatus.BLOCKED and cache.get("captions", "third|en") is None
