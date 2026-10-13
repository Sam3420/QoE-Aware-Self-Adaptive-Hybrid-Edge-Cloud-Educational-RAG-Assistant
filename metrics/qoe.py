"""Quality of Experience: QoE = wEQ·EQ + wAQ·AQ + wNQ·NQ + wUX·UX, every term normalised to [0, 1].

    EQ = α1·LG + α2·PT + α3·RA + α4·LS     (learning gain, post-test, recommendation acceptance, satisfaction)
    AQ = β1·RQ + β2·LQ + β3·SQ + β4·TQ     (RAG, LLM response, speech-to-text, text-to-speech quality)
    NQ = γ1·PL + γ2·NR + γ3·RU + γ4·AV     (latency quality, network reliability, utilisation quality, availability)
    UX = δ1·SR + δ2·FS + δ3·CR             (star rating, feedback sentiment, completion rate)

Raw measurements (QoERaw) are converted by `normalize_qoe` with explicit bounds (QoEBounds); cost
metrics are inverted there, so every normalised metric is "higher = better". Missing metrics are None
and handled by the configured policy (metrics.weights.aggregate) inside each group and again across
groups. The result always lists coverage and missing inputs, so it never claims more evidence than it has.
"""
from pydantic import BaseModel, Field, field_validator

from metrics.normalization import benefit, cost, fraction, signed, stars, utilization_band
from metrics.weights import Aggregate, GROUP_KEYS, MissingPolicy, WeightSet, aggregate

GROUPS = ("eq", "aq", "nq", "ux")
TOP_KEY = {"eq": "EQ", "aq": "AQ", "nq": "NQ", "ux": "UX"}


class QoERaw(BaseModel):
    """Measurements before normalisation (None = not measured)."""

    learning_gain: float | None = None  # Hake gain (post − pre)/(1 − pre), may be negative
    post_test: float | None = None  # 0–1
    recommendation_acceptance: float | None = None  # 0–1
    learner_satisfaction_stars: float | None = None  # 1–5 (ratings of learning resources)
    retrieval_score: float | None = None  # mean CRAG top retrieval similarity
    llm_quality: float | None = None  # 0–1 from an external evaluator; none exists in the project
    stt_quality: float | None = None  # 0–1; speech-to-text is not implemented
    tts_quality: float | None = None  # 0–1; text-to-speech is not implemented
    latency_ms: float | None = None  # mean end-to-end answer latency
    network_reliability: float | None = None  # successful network operations / attempted, 0–1
    utilization: float | None = None  # peak of CPU and memory utilisation, 0–1
    availability: float | None = None  # answers produced by an LLM route / questions, 0–1
    star_rating: float | None = None  # 1–5 session rating
    sentiment: float | None = None  # −1..1 (lexicon estimate)
    completion: float | None = None  # 0–1


class QoEBounds(BaseModel):
    """Normalisation bounds (from config)."""

    latency_good_ms: float = 1500.0
    latency_bad_ms: float = 15000.0
    retrieval_floor: float = 0.40
    retrieval_ceil: float = 0.85
    utilization_target: float = 0.70
    utilization_saturation: float = 0.95


class QoEMetrics(BaseModel):
    """Normalised metrics in [0, 1], higher = better; None = unavailable."""

    LG: float | None = None
    PT: float | None = None
    RA: float | None = None
    LS: float | None = None
    RQ: float | None = None
    LQ: float | None = None
    SQ: float | None = None
    TQ: float | None = None
    PL: float | None = None
    NR: float | None = None
    RU: float | None = None
    AV: float | None = None
    SR: float | None = None
    FS: float | None = None
    CR: float | None = None

    @field_validator("*")
    @classmethod
    def _unit_interval(cls, value: float | None) -> float | None:
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError("normalised metrics must be in [0, 1]")
        return value


def normalize_qoe(raw: QoERaw, bounds: QoEBounds) -> QoEMetrics:
    """Convert raw measurements to [0, 1] metrics (latency inverted; utilisation as a health band)."""
    return QoEMetrics(
        LG=signed(raw.learning_gain),  # gain −1 → 0, 0 → 0.5, 1 → 1
        PT=fraction(raw.post_test),
        RA=fraction(raw.recommendation_acceptance),
        LS=stars(raw.learner_satisfaction_stars),
        RQ=benefit(raw.retrieval_score, bounds.retrieval_floor, bounds.retrieval_ceil),
        LQ=fraction(raw.llm_quality),
        SQ=fraction(raw.stt_quality),
        TQ=fraction(raw.tts_quality),
        PL=cost(raw.latency_ms, bounds.latency_good_ms, bounds.latency_bad_ms),
        NR=fraction(raw.network_reliability),
        RU=utilization_band(raw.utilization, bounds.utilization_target, bounds.utilization_saturation),
        AV=fraction(raw.availability),
        SR=stars(raw.star_rating),
        FS=signed(raw.sentiment),
        CR=fraction(raw.completion),
    )


class QoEResult(BaseModel):
    """Overall QoE and every intermediate score."""

    qoe: float | None
    components: dict[str, float | None]  # EQ, AQ, NQ, UX
    groups: dict[str, Aggregate]  # per-group detail (used, missing, coverage)
    top: Aggregate
    coverage: float = Field(ge=0.0, le=1.0)  # Σ_groups wG · coverage_G: share of all weight backed by data
    missing_groups: list[str]
    policy: str


def compute_qoe(metrics: QoEMetrics, weights: WeightSet, policy: MissingPolicy = "renormalize") -> QoEResult:
    """Aggregate normalised metrics into EQ, AQ, NQ, UX and QoE with the missing-data policy."""
    values = metrics.model_dump()
    groups = {g: aggregate({k: values[k] for k in GROUP_KEYS[g]}, weights.get(g), policy) for g in GROUPS}
    components = {TOP_KEY[g]: groups[g].score for g in GROUPS}
    # A group with no data at all is unavailable at the top level too (never scored as 0).
    top = aggregate(components, weights.get("qoe"), policy)
    top_w = weights.get("qoe")
    coverage = round(sum(top_w[TOP_KEY[g]] * groups[g].coverage for g in GROUPS), 6)
    return QoEResult(qoe=top.score, components=components, groups=groups, top=top, coverage=min(coverage, 1.0),
                     missing_groups=[TOP_KEY[g] for g in GROUPS if groups[g].score is None], policy=policy)
