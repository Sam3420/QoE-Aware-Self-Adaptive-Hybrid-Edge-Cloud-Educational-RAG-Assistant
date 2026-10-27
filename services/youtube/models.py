"""Internal data models for search candidates, transcripts and recommendations (validated with pydantic).

Every optional field is None when YouTube did not return it; None means "unknown", never "bad".
"""
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class VideoCandidate(BaseModel):
    """One video from search.list, enriched with videos.list metadata when available."""

    video_id: str
    search_rank: int | None = None  # position in the search results (0 = first)
    title: str = ""
    description: str = ""
    channel_id: str | None = None
    channel_title: str | None = None
    published_at: datetime | None = None
    thumbnail_url: str | None = None
    category_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    live_broadcast_content: str | None = None  # "none" | "live" | "upcoming"
    default_language: str | None = None
    default_audio_language: str | None = None
    # contentDetails
    duration_s: int | None = None
    caption_flag: bool | None = None  # contentDetails.caption: True if the video has manual captions
    definition: str | None = None
    region_allowed: list[str] | None = None
    region_blocked: list[str] | None = None
    # statistics (absent when hidden by the owner)
    view_count: int | None = None
    like_count: int | None = None
    comment_count: int | None = None
    # status
    privacy_status: str | None = None
    upload_status: str | None = None
    embeddable: bool | None = None
    made_for_kids: bool | None = None
    has_details: bool = False  # True once videos.list data was merged in

    @property
    def url(self) -> str:
        """Watch URL."""
        return f"https://www.youtube.com/watch?v={self.video_id}"


class TranscriptStatus(StrEnum):
    """Why a transcript is or is not available; every value except OK keeps the video in ranking."""

    OK = "ok"
    NOT_ATTEMPTED = "not_attempted"  # outside the transcript budget
    DISABLED_BY_CONFIG = "disabled_by_config"
    NO_TRANSCRIPT = "no_transcript"  # no track in the preferred languages
    TRANSCRIPTS_DISABLED = "transcripts_disabled"  # the owner disabled captions
    VIDEO_UNAVAILABLE = "video_unavailable"
    AGE_RESTRICTED = "age_restricted"  # needs authentication; not bypassed
    BLOCKED = "blocked"  # IP or request blocking by YouTube; retrieval stops for the run
    FAILED = "failed"  # network error, parse error, library incompatibility


class TranscriptSegment(BaseModel):
    """One caption segment."""

    start: float = Field(ge=0.0)
    duration: float = Field(ge=0.0)
    text: str


class TranscriptRecord(BaseModel):
    """Normalised transcript (or the reason there is none) for one video."""

    video_id: str
    status: TranscriptStatus
    language_code: str | None = None
    language: str | None = None
    caption_type: str | None = None  # "manual" | "auto" | "<manual|auto>-translated from <lang>" (YouTube's translation)
    segments: list[TranscriptSegment] = Field(default_factory=list)
    retrieved_at: datetime | None = None
    error: str | None = None
    from_cache: bool = False
    truncated: bool = False

    @property
    def text(self) -> str:
        """Plain transcript text."""
        return " ".join(s.text for s in self.segments)


class TranscriptChunk(BaseModel):
    """A window of transcript text with its time range."""

    video_id: str
    index: int
    start: float
    end: float
    text: str
    language_code: str | None = None


class EvidenceExcerpt(BaseModel):
    """A transcript passage that supports a relevance claim."""

    start: float
    end: float
    text: str
    similarity: float = Field(ge=0.0, le=1.0)  # normalised


class TranscriptEvidence(BaseModel):
    """What the transcript shows about the learning objective (only built from real transcript text)."""

    relevance: float = Field(ge=0.0, le=1.0)
    excerpts: list[EvidenceExcerpt] = Field(default_factory=list)
    concepts_covered: dict[str, float] = Field(default_factory=dict)  # concept → first timestamp (s)
    concepts_missing: list[str] = Field(default_factory=list)
    example_timestamps: list[float] = Field(default_factory=list)
    step_timestamps: list[float] = Field(default_factory=list)
    prerequisite_timestamps: list[float] = Field(default_factory=list)
    on_topic_seconds: float = 0.0
    n_chunks: int = 0


class Recommendation(BaseModel):
    """One ranked video, as returned to the UI/API."""

    rank: int
    video_id: str
    title: str
    description: str
    thumbnail_url: str | None
    channel_title: str | None
    duration_s: int | None
    published_at: datetime | None
    url: str
    embeddable: bool | None
    score: float
    score_breakdown: dict[str, float | None]  # feature → normalised value (None = unavailable)
    evidence_level: str  # "transcript" | "metadata"
    evidence_coverage: float  # share of ranking weight backed by available features
    matched_objective: str
    explanation: str
    transcript_status: TranscriptStatus
    caption_language: str | None = None
    caption_type: str | None = None
    transcript_evidence: TranscriptEvidence | None = None
    item_id: str | None = None  # persisted row id, set once stored


class FilteredVideo(BaseModel):
    """A candidate removed by a hard constraint, with the reason."""

    video_id: str
    title: str
    reason: str


class StageTimings(BaseModel):
    """Latency of each pipeline stage in ms (None when the stage did not run)."""

    search_ms: float | None = None
    metadata_ms: float | None = None
    transcript_ms: float | None = None
    embedding_ms: float | None = None
    ranking_ms: float | None = None
    total_ms: float | None = None


class PipelineMetrics(BaseModel):
    """Operational instrumentation of one recommendation run."""

    timings: StageTimings = Field(default_factory=StageTimings)
    api_calls: int = 0
    api_failures: int = 0
    quota_units: int = 0
    api_bytes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    candidates_found: int = 0
    candidates_filtered: int = 0
    transcripts_attempted: int = 0
    transcripts_ok: int = 0
    transcripts_from_cache: int = 0
    embeddings: int = 0
    cpu: float | None = None
    mem: float | None = None

    def rates(self) -> dict[str, float | None]:
        """Derived rates; None when the denominator is zero."""
        def ratio(a: int, b: int) -> float | None:
            return round(a / b, 4) if b else None

        return {
            "api_failure_rate": ratio(self.api_failures, self.api_calls),
            "transcript_success_rate": ratio(self.transcripts_ok, self.transcripts_attempted),
            "candidate_filtering_rate": ratio(self.candidates_filtered, self.candidates_found),
            "cache_hit_rate": ratio(self.cache_hits, self.cache_hits + self.cache_misses),
        }


class RecommendationResult(BaseModel):
    """Everything one pipeline run produced."""

    query: str
    objective: str
    recommendations: list[Recommendation]
    filtered: list[FilteredVideo]
    metrics: PipelineMetrics
    errors: list[str] = Field(default_factory=list)
    status: str = "ok"  # "ok" | "empty" | "degraded" | "failed"
    extra: dict[str, Any] = Field(default_factory=dict)
