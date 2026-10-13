"""Learner-aware ranking: hard filters, normalised features, weighted score with explicit missing-data rules.

Score of one video (every feature in [0, 1], None = unavailable):

    observed  = Σ_{f available} w_f · x_f  /  Σ_{f available} w_f
    coverage  = Σ_{f available} w_f        /  Σ_f w_f
    score     = coverage · observed + (1 − coverage) · prior          (prior = REC_MISSING_FEATURE_PRIOR)

A missing feature therefore pulls the score toward a neutral prior instead of toward 0: a missing
transcript or hidden statistics is "unknown", not "bad". It also means a metadata-only video whose
observed score equals a transcript-backed one ranks slightly lower when that score is above the
prior. That is the deliberate evidence-strength discount (stated in each recommendation's evidence_level
and evidence_coverage).

This module is pure (no I/O, no config import) so it is fully unit-testable.
"""
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, field_validator

from services.youtube.models import FilteredVideo, TranscriptEvidence, TranscriptRecord, TranscriptStatus, VideoCandidate

FEATURES: tuple[str, ...] = ("topic_similarity", "transcript_relevance", "difficulty_match", "language_match",
                             "duration_match", "caption_match", "historical_outcome", "popularity")
_BEGINNER = re.compile(r"\b(basics?|beginners?|introduction|intro|simple|simply|easy|for kids|made easy|"
                       r"class [1-8]|grade [1-8]|explained)\b", re.IGNORECASE)
_INTERMEDIATE = re.compile(r"\b(class (?:9|10)|grade (?:9|10)|high school|ncert|cbse|gcse)\b", re.IGNORECASE)
_ADVANCED = re.compile(r"\b(advanced|in[- ]depth|derivation|university|college|lecture|graduate|"
                       r"class 1[12]|grade 1[12]|a[- ]level|biochemistry|mechanism)\b", re.IGNORECASE)
LANGUAGE_CODES = {"english": "en", "hindi": "hi"}


class RankingWeights(BaseModel):
    """Validated, configurable feature weights (normalised to sum to 1)."""

    weights: dict[str, float]

    @field_validator("weights")
    @classmethod
    def _valid(cls, value: dict[str, float]) -> dict[str, float]:
        unknown = set(value) - set(FEATURES)
        if unknown:
            raise ValueError(f"unknown ranking features: {sorted(unknown)}; allowed: {list(FEATURES)}")
        if any((not math.isfinite(w)) or w < 0 for w in value.values()):
            raise ValueError("ranking weights must be finite and non-negative")
        total = sum(value.values())
        if total <= 0:
            raise ValueError("ranking weights must not all be zero")
        return {f: float(value.get(f, 0.0)) / total for f in FEATURES}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float]) -> "RankingWeights":
        return cls(weights=dict(mapping))


@dataclass(frozen=True)
class LearnerSignals:
    """What the ranking knows about the learner (None = unknown)."""

    language_code: str | None
    competency_num: float | None  # beginner 0, intermediate 0.5, advanced 1
    duration_min_s: int
    duration_max_s: int


@dataclass
class ScoredVideo:
    """A candidate with its features, combined score and evidence."""

    candidate: VideoCandidate
    features: dict[str, float | None]
    score: float = 0.0
    coverage: float = 0.0
    transcript: TranscriptRecord | None = None
    evidence: TranscriptEvidence | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def evidence_level(self) -> str:
        return "transcript" if self.features.get("transcript_relevance") is not None else "metadata"


# ---- hard constraints -----------------------------------------------------------------------------

def hard_filter(candidates: Iterable[VideoCandidate], *, region_code: str | None, min_duration_s: int,
                require_embeddable: bool, require_language: str | None, excluded: Mapping[str, str],
                ) -> tuple[list[VideoCandidate], list[FilteredVideo]]:
    """Drop duplicates and videos that cannot or should not be shown; unknown values never filter."""
    kept, filtered, seen = [], [], set()
    for c in candidates:
        reason = None
        if c.video_id in seen:
            reason = "duplicate"
        elif c.video_id in excluded:
            reason = f"already {excluded[c.video_id]} by this learner"
        elif c.privacy_status not in (None, "public"):
            reason = f"privacy status {c.privacy_status}"
        elif c.upload_status not in (None, "processed"):
            reason = f"upload status {c.upload_status}"
        elif c.live_broadcast_content in ("live", "upcoming"):
            reason = f"live broadcast ({c.live_broadcast_content})"
        elif c.duration_s is not None and c.duration_s < min_duration_s:
            reason = f"shorter than {min_duration_s}s"
        elif region_code and c.region_blocked and region_code in c.region_blocked:
            reason = f"blocked in {region_code}"
        elif region_code and c.region_allowed is not None and region_code not in c.region_allowed:
            reason = f"not available in {region_code}"
        elif require_embeddable and c.embeddable is False:
            reason = "embedding disabled by the owner"
        elif require_language:
            lang = video_language(c, None)
            if lang is not None and lang != require_language:
                reason = f"language {lang} ≠ required {require_language}"
        seen.add(c.video_id)
        if reason:
            filtered.append(FilteredVideo(video_id=c.video_id, title=c.title, reason=reason))
        else:
            kept.append(c)
    return kept, filtered


# ---- features ------------------------------------------------------------------------------------

def _primary(code: str | None) -> str | None:
    return code.split("-")[0].lower() if code else None


def video_language(candidate: VideoCandidate, transcript: TranscriptRecord | None) -> str | None:
    """Best-known spoken language: defaultAudioLanguage, else a manual transcript's, else defaultLanguage."""
    if candidate.default_audio_language:
        return _primary(candidate.default_audio_language)
    if transcript and transcript.status == TranscriptStatus.OK and transcript.caption_type == "manual":
        return _primary(transcript.language_code)
    if transcript and transcript.status == TranscriptStatus.OK:
        return _primary(transcript.language_code)  # auto captions follow the spoken language
    return _primary(candidate.default_language)


def estimate_video_level(text: str) -> float | None:
    """Heuristic level from wording ('basics' 0, 'class 10' 0.5, 'derivation' 1); None if no cue."""
    levels = [0.0] * len(_BEGINNER.findall(text)) + [0.5] * len(_INTERMEDIATE.findall(text)) + \
        [1.0] * len(_ADVANCED.findall(text))
    return round(sum(levels) / len(levels), 3) if levels else None


def difficulty_match(video_level: float | None, learner_level: float | None) -> float | None:
    if video_level is None or learner_level is None:
        return None
    return round(1.0 - abs(video_level - learner_level), 4)


def language_match(candidate: VideoCandidate, transcript: TranscriptRecord | None, learner_lang: str | None) -> float | None:
    if not learner_lang:
        return None
    lang = video_language(candidate, transcript)
    return None if lang is None else float(lang == learner_lang)


def duration_match(duration_s: int | None, lo: int, hi: int) -> float | None:
    """1 inside [lo, hi]; falls linearly to 0 at 0 s below and at 2·hi above."""
    if duration_s is None:
        return None
    if lo <= duration_s <= hi:
        return 1.0
    if duration_s < lo:
        return round(duration_s / lo, 4) if lo else 1.0
    return round(max(0.0, 1.0 - (duration_s - hi) / hi), 4)


def caption_match(candidate: VideoCandidate, transcript: TranscriptRecord | None) -> float | None:
    """1 if a usable transcript/caption exists, 0 only when retrieval confirmed none, else unknown.

    contentDetails.caption = false only means "no manual captions" (auto captions may exist), so it
    is not treated as 0.
    """
    if transcript is not None:
        if transcript.status == TranscriptStatus.OK:
            return 1.0
        if transcript.status in (TranscriptStatus.NO_TRANSCRIPT, TranscriptStatus.TRANSCRIPTS_DISABLED):
            return 0.0
    if candidate.caption_flag:
        return 1.0
    return None


def popularity_scores(candidates: Sequence[VideoCandidate]) -> dict[str, float | None]:
    """Secondary signal: log views min-max scaled within this candidate set, blended with like ratio."""
    logs = {c.video_id: math.log10(c.view_count + 1) for c in candidates if c.view_count is not None}
    if not logs:
        return {c.video_id: None for c in candidates}
    lo, hi = min(logs.values()), max(logs.values())
    out: dict[str, float | None] = {}
    for c in candidates:
        if c.video_id not in logs:
            out[c.video_id] = None
            continue
        views = (logs[c.video_id] - lo) / (hi - lo) if hi > lo else 0.5
        if c.like_count is not None and c.view_count:
            like_ratio = min(c.like_count / c.view_count / 0.05, 1.0)  # 5% likes/views counts as excellent
            out[c.video_id] = round(0.7 * views + 0.3 * like_ratio, 4)
        else:
            out[c.video_id] = round(views, 4)
    return out


def historical_outcome(history: Mapping[str, Any] | None) -> float | None:
    """Past feedback on this video: smoothed accept ratio and mean rating/5; None without any feedback."""
    if not history:
        return None
    parts = []
    accepted, dismissed = int(history.get("accepted", 0)), int(history.get("dismissed", 0))
    if accepted + dismissed:
        parts.append((accepted + 1) / (accepted + dismissed + 2))  # Beta(1, 1) prior
    ratings = [r for r in history.get("ratings", []) if isinstance(r, (int, float))]
    if ratings:
        parts.append(sum(ratings) / len(ratings) / 5.0)
    return round(sum(parts) / len(parts), 4) if parts else None


def combine(features: Mapping[str, float | None], weights: RankingWeights, prior: float) -> tuple[float, float]:
    """Return (score, coverage) using the missing-feature rule in the module docstring."""
    w = weights.weights
    available = {f: v for f, v in features.items() if v is not None and w.get(f, 0) > 0}
    coverage = sum(w[f] for f in available)
    if coverage <= 0:
        return prior, 0.0
    observed = sum(w[f] * v for f, v in available.items()) / coverage
    return round(coverage * observed + (1 - coverage) * prior, 4), round(coverage, 4)


# ---- explanation ---------------------------------------------------------------------------------

def mmss(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 \
        else f"{seconds // 60:02d}:{seconds % 60:02d}"


def explain(video: ScoredVideo, learner: LearnerSignals) -> str:
    """Short, grounded explanation: transcript claims cite timestamps; metadata claims say so."""
    f, c, parts = video.features, video.candidate, []
    ev = video.evidence
    if ev is not None and video.transcript is not None:
        kind = video.transcript.caption_type or "unknown-type"
        if ev.excerpts:
            parts.append(f"Transcript ({video.transcript.language_code}, {kind} captions) addresses the objective at "
                         f"{mmss(ev.excerpts[0].start)}: “{ev.excerpts[0].text[:110].rstrip()}…”.")
        total = len(ev.concepts_covered) + len(ev.concepts_missing)
        if total:
            covered = ", ".join(f"{k} at {mmss(t)}" for k, t in list(ev.concepts_covered.items())[:3])
            parts.append(f"Covers {len(ev.concepts_covered)} of {total} target concepts" +
                         (f" ({covered})." if covered else "."))
        cues = [n for n, ts in (("examples", ev.example_timestamps), ("step-by-step explanation", ev.step_timestamps))
                if ts]
        if cues:
            parts.append(f"Includes {' and '.join(cues)} in the relevant parts.")
    else:
        reason = video.transcript.status.value.replace("_", " ") if video.transcript else "not attempted"
        sim = f.get("topic_similarity")
        parts.append(f"Ranked on title and description only{f' (match {sim:.2f})' if sim is not None else ''}; "
                     f"no transcript evidence ({reason}).")
    if c.duration_s is not None:
        fits = "within" if f.get("duration_match") == 1.0 else "outside"
        parts.append(f"{round(c.duration_s / 60)} min, {fits} the preferred "
                     f"{learner.duration_min_s // 60}–{learner.duration_max_s // 60} min.")
    level = video.notes.get("video_level")
    if level is not None and f.get("difficulty_match") is not None:
        name = "beginner" if level < 0.34 else "intermediate" if level < 0.67 else "advanced"
        parts.append(f"Wording suggests {name} level (heuristic).")
    return " ".join(parts)
