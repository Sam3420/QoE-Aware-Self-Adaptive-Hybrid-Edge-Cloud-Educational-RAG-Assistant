"""Feed the concept tracker from graded pre/post-test answers (live after each test, and the one-off backfill).

Each test item carries a free-text ``concept`` label from question generation. The topic's vocabulary matcher maps it
to a concept id (name/alias, then embedding similarity); the answer is recorded as one attempt on that concept with
``source_ref = "<assessment id>:<item index>"``, so recording a test twice changes nothing. Items whose label maps to
no concept are reported, not guessed.
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.db.repositories import ConceptTrackerRepository
from app.services.mcq import normalize_answer
from knowledge_prep.concepts import ConceptMatcher, load_vocabulary

SOURCES = {"pre": "pre-test", "post": "post-test"}


@dataclass
class TrackReport:
    """What recording one or more assessments did."""

    assessments: int = 0
    items: int = 0
    recorded: int = 0  # new attempts
    already: int = 0  # attempts recorded before (idempotent re-run)
    unmapped: Counter = field(default_factory=Counter)  # concept label -> items it could not be mapped
    methods: Counter = field(default_factory=Counter)  # exact / alias / embedding
    skipped_topics: Counter = field(default_factory=Counter)  # topic -> assessments without a vocabulary
    vocabulary_status: dict[str, str] = field(default_factory=dict)  # topic -> draft / approved
    matches: dict[str, tuple[str | None, str, float | None]] = field(default_factory=dict)  # label -> match

    def merge(self, other: "TrackReport") -> None:
        for name in ("assessments", "items", "recorded", "already"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in ("unmapped", "methods", "skipped_topics"):
            getattr(self, name).update(getattr(other, name))
        self.vocabulary_status.update(other.vocabulary_status)
        self.matches.update(other.matches)


class AssessmentTracker:
    """Records assessments into the tracker, caching one matcher per topic."""

    def __init__(self, embedder: Any = None) -> None:
        self.embedder = embedder
        self._matchers: dict[str, ConceptMatcher | None] = {}

    def matcher(self, topic_id: str) -> ConceptMatcher | None:
        if topic_id not in self._matchers:
            vocabulary = load_vocabulary(topic_id)
            self._matchers[topic_id] = ConceptMatcher(vocabulary, self.embedder) if vocabulary else None
        return self._matchers[topic_id]

    def record(self, repo: ConceptTrackerRepository, assessment: dict[str, Any]) -> TrackReport:
        """Record every graded item of one assessment as an attempt on its concept."""
        report = TrackReport(assessments=1)
        matcher = self.matcher(assessment["topic_id"])
        if matcher is None:
            report.skipped_topics[assessment["topic_id"]] += 1
            return report
        report.vocabulary_status[assessment["topic_id"]] = matcher.vocabulary.status
        source = SOURCES.get(assessment["type"], f"{assessment['type']}-test")
        answers = assessment.get("answers") or []
        for index, item in enumerate(assessment.get("items") or []):
            report.items += 1
            label = str(item.get("concept") or "")
            match = matcher.match(label, context=str(item.get("question") or ""))
            report.matches[label] = (match.concept_id, match.reason or match.method, match.similarity)
            if match.concept_id is None:
                report.unmapped[label] += 1
                continue
            report.methods[match.method] += 1
            answer = answers[index] if index < len(answers) else None
            correct = normalize_answer(answer) == item.get("answer_index")
            before = repo.get_mastery(assessment["student_id"], assessment["topic_id"], match.concept_id)
            after = repo.record_attempt(assessment["student_id"], assessment["topic_id"], match.concept_id, correct,
                                        source=source, source_ref=f"{assessment['id']}:{index}",
                                        occurred_at=assessment.get("created_at"))
            if after["attempts"] > before["attempts"]:
                report.recorded += 1
            else:
                report.already += 1
        return report


def track_assessment(services: Any, assessment_id: str, tracker: AssessmentTracker | None = None) -> TrackReport:
    """Record one just-scored assessment (called by the pre/post-test nodes)."""
    from app.db.repositories import AdaptationRepository

    tracker = tracker or AssessmentTracker(getattr(services.crag.store, "embedder", None))
    with services.app._db() as db:
        assessment = AdaptationRepository(db).get_assessment(assessment_id)
        if assessment is None:
            return TrackReport()
        return tracker.record(ConceptTrackerRepository(db), assessment)
