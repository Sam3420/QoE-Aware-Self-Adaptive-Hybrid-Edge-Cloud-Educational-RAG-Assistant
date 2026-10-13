from datetime import datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.models import Experience, ExperienceTransition, ExperienceVector, WeightSetRecord


class AdaptiveLearningRepository:
    """Transitions, FAISS id mapping and weight sets. Returns plain dicts."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # Transitions

    def get_transition_by_session(self, session_id: str) -> dict[str, Any] | None:
        row = self.session.scalars(select(ExperienceTransition).where(ExperienceTransition.session_id == session_id)).first()
        return self._record(row.experience_id) if row else None

    def add_transition(self, *, experience_id: str, session_id: str, action: dict[str, Any],
                       prev_state: dict[str, Any], next_state: dict[str, Any], scores: dict[str, Any],
                       utility_before: float, utility_after: float, predicted_utility: float | None,
                       baseline_used: bool, weights_version: str) -> None:
        self.session.add(ExperienceTransition(
            experience_id=experience_id, session_id=session_id, action=action, prev_state=prev_state,
            next_state=next_state, scores=scores, utility_before=utility_before, utility_after=utility_after,
            predicted_utility=predicted_utility, baseline_used=baseline_used, weights_version=weights_version))
        self.session.flush()

    def records(self, experience_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Full experience records (experiences + transitions) by id; ids without a transition are skipped."""
        if not experience_ids:
            return {}
        rows = self.session.execute(
            select(Experience, ExperienceTransition)
            .join(ExperienceTransition, ExperienceTransition.experience_id == Experience.id)
            .where(Experience.id.in_(experience_ids)))
        return {e.id: _record_dict(e, t) for e, t in rows}

    def _record(self, experience_id: str) -> dict[str, Any] | None:
        return self.records([experience_id]).get(experience_id)

    def last_transition_for_student(self, student_id: str) -> dict[str, Any] | None:
        """The learner's most recent full transition (its s_next is the next session's s)."""
        row = self.session.execute(
            select(Experience, ExperienceTransition)
            .join(ExperienceTransition, ExperienceTransition.experience_id == Experience.id)
            .where(Experience.student_id == student_id)
            .order_by(Experience.created_at.desc()).limit(1)).first()
        return _record_dict(*row) if row else None

    def list_transitions(self, limit: int | None = None) -> list[dict[str, Any]]:
        statement = (select(Experience, ExperienceTransition)
                     .join(ExperienceTransition, ExperienceTransition.experience_id == Experience.id)
                     .order_by(Experience.created_at.desc()))
        if limit is not None:
            statement = statement.limit(limit)
        return [_record_dict(e, t) for e, t in self.session.execute(statement)]

    # FAISS mapping

    def add_vector(self, experience_id: str, vector: list[float], embedding_version: str) -> int:
        row = ExperienceVector(experience_id=experience_id, vector=vector, embedding_version=embedding_version)
        self.session.add(row)
        self.session.flush()
        return row.faiss_id

    def list_vectors(self, embedding_version: str) -> list[tuple[int, str, list[float]]]:
        statement = (select(ExperienceVector).where(ExperienceVector.embedding_version == embedding_version)
                     .order_by(ExperienceVector.faiss_id))
        return [(v.faiss_id, v.experience_id, v.vector) for v in self.session.scalars(statement)]

    def experience_ids_for(self, faiss_ids: list[int]) -> dict[int, str]:
        if not faiss_ids:
            return {}
        rows = self.session.execute(select(ExperienceVector.faiss_id, ExperienceVector.experience_id)
                                    .where(ExperienceVector.faiss_id.in_(faiss_ids)))
        return {fid: eid for fid, eid in rows}

    def replace_vectors(self, items: list[tuple[str, list[float]]], embedding_version: str) -> None:
        """Re-embed every transition (used when the embedding version changes)."""
        for row in self.session.scalars(select(ExperienceVector)):
            self.session.delete(row)
        self.session.flush()
        for experience_id, vector in items:
            self.add_vector(experience_id, vector, embedding_version)

    # Weight sets

    def save_weight_set(self, *, group: str, method: str, weights: dict[str, float], baseline: dict[str, float],
                        details: dict[str, Any], n_observations: int, n_used: int, window_start: datetime | None,
                        window_end: datetime | None, data_version: str, activate: bool) -> dict[str, Any]:
        if activate:
            self.session.execute(update(WeightSetRecord).where(WeightSetRecord.group == group).values(active=False))
        row = WeightSetRecord(group=group, method=method, weights=weights, baseline=baseline, details=details,
                              n_observations=n_observations, n_used=n_used, window_start=window_start,
                              window_end=window_end, data_version=data_version, active=activate)
        self.session.add(row)
        self.session.flush()
        return _weight_dict(row)

    def active_weight_sets(self) -> dict[str, dict[str, Any]]:
        rows = self.session.scalars(select(WeightSetRecord).where(WeightSetRecord.active.is_(True)))
        return {r.group: _weight_dict(r) for r in rows}

    def list_weight_sets(self, group: str | None = None) -> list[dict[str, Any]]:
        statement = select(WeightSetRecord).order_by(WeightSetRecord.created_at.desc())
        if group:
            statement = statement.where(WeightSetRecord.group == group)
        return [_weight_dict(r) for r in self.session.scalars(statement)]


def _record_dict(e: Experience, t: ExperienceTransition) -> dict[str, Any]:
    return {
        "id": e.id, "student_id": e.student_id, "session_id": e.session_id, "topic_id": e.topic_id,
        "state": e.state, "action": t.action, "policy_id": e.policy_id, "reward": e.reward,
        "prev_state": t.prev_state, "next_state": t.next_state,
        "qoe_before": e.qoe_before, "qoe_after": e.qoe_after, "sre_before": e.sre_before, "sre_after": e.sre_after,
        "utility_before": t.utility_before, "utility_after": t.utility_after, "predicted_utility": t.predicted_utility,
        "scores": t.scores, "baseline_used": t.baseline_used, "weights_version": t.weights_version,
        "created_at": e.created_at,
    }


def _weight_dict(r: WeightSetRecord) -> dict[str, Any]:
    return {"id": r.id, "group": r.group, "method": r.method, "weights": r.weights, "baseline": r.baseline,
            "details": r.details, "n_observations": r.n_observations, "n_used": r.n_used,
            "window_start": r.window_start, "window_end": r.window_end, "data_version": r.data_version,
            "active": r.active, "created_at": r.created_at}
