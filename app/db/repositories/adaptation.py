from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import POLICY_SEED
from app.db.errors import ReferencedRecordNotFoundError
from app.db.models import (
    Assessment,
    Experience,
    LearnerProfile,
    LearningSession,
    Policy,
    TopicQuestionBank,
)
from app.domain.enums import AssessmentType
from app.domain.models import AssessmentCreate, ExperienceCreate, ExperienceState

# Order of the kNN state vector returned by get_experiences().
EXPERIENCE_STATE_KEYS: tuple[str, ...] = tuple(ExperienceState.model_fields)


class AdaptationRepository:
    """Persistence for assessments, policies, experiences, and topic question banks.

    Every public method returns plain dicts (never ORM objects) so results can be
    placed directly into the agent state.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    # Assessments

    def create_assessment(self, data: AssessmentCreate) -> dict[str, Any]:
        """Store a pre/post assessment attempt and return it as a dict."""
        self._require_learner_and_session(data.student_id, data.session_id)
        assessment = Assessment(
            session_id=data.session_id,
            student_id=data.student_id,
            topic_id=data.topic_id,
            type=data.type.value,
            items=data.items,
            answers=data.answers,
            score=data.score,
            competency=data.competency.value,
        )
        self.session.add(assessment)
        self.session.flush()
        return _assessment_to_dict(assessment)

    def get_latest_assessment(
        self,
        student_id: str,
        topic_id: str,
        type: AssessmentType | str,
    ) -> dict[str, Any] | None:
        """Return the most recent assessment of ``type`` for a learner and topic, or None."""
        statement = (
            select(Assessment)
            .where(
                Assessment.student_id == student_id,
                Assessment.topic_id == topic_id,
                Assessment.type == AssessmentType(type).value,
            )
            .order_by(Assessment.created_at.desc())
            .limit(1)
        )
        assessment = self.session.scalars(statement).first()
        return _assessment_to_dict(assessment) if assessment is not None else None

    def has_pre_score(self, student_id: str, topic_id: str) -> bool:
        """Return True if the learner already has a pre-test score for the topic."""
        return self.get_latest_assessment(student_id, topic_id, AssessmentType.PRE) is not None

    # Policies

    def seed_policies(self) -> int:
        """Insert any missing predefined policies; safe to call repeatedly. Returns rows added."""
        existing = set(self.session.scalars(select(Policy.policy_id)))
        added = 0
        for policy in POLICY_SEED:
            if policy["policy_id"] not in existing:
                self.session.add(Policy(**policy))
                added += 1
        self.session.flush()
        return added

    def list_policies(self) -> list[dict[str, Any]]:
        """Return all policies ordered by policy_id."""
        statement = select(Policy).order_by(Policy.policy_id)
        return [_policy_to_dict(policy) for policy in self.session.scalars(statement)]

    def get_policy(self, policy_id: str) -> dict[str, Any] | None:
        """Return one policy by id (e.g. ``"P1"``), or None."""
        policy = self.session.get(Policy, policy_id)
        return _policy_to_dict(policy) if policy is not None else None

    def get_default_policy(self) -> dict[str, Any] | None:
        """Return the policy flagged ``is_default``, or None if policies are not seeded."""
        statement = select(Policy).where(Policy.is_default.is_(True)).order_by(Policy.policy_id)
        policy = self.session.scalars(statement).first()
        return _policy_to_dict(policy) if policy is not None else None

    # Experiences

    def insert_experience(self, data: ExperienceCreate) -> dict[str, Any]:
        """Store one session experience and return it in get_experiences() shape."""
        self._require_learner_and_session(data.student_id, data.session_id)
        if self.session.get(Policy, data.policy_id) is None:
            raise ReferencedRecordNotFoundError(f"Policy {data.policy_id!r} does not exist.")
        experience = Experience(
            student_id=data.student_id,
            session_id=data.session_id,
            topic_id=data.topic_id,
            state=data.state.model_dump(),
            policy_id=data.policy_id,
            qoe_before=data.qoe_before,
            qoe_after=data.qoe_after,
            sre_before=data.sre_before,
            sre_after=data.sre_after,
            reward=data.reward,
        )
        self.session.add(experience)
        self.session.flush()
        return _experience_to_dict(experience)

    def get_experiences(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return experiences newest first, shaped for kNN.

        Each dict has ``state`` (dict), ``state_vector`` (values in
        EXPERIENCE_STATE_KEYS order; ``last_qoe`` may be None), ``policy_id`` and ``reward``.
        """
        statement = select(Experience).order_by(Experience.created_at.desc())
        if limit is not None:
            statement = statement.limit(limit)
        return [_experience_to_dict(experience) for experience in self.session.scalars(statement)]

    def get_last_qoe(self, student_id: str) -> float | None:
        """Return ``qoe_after`` of the learner's most recent experience, or None."""
        statement = (
            select(Experience.qoe_after)
            .where(Experience.student_id == student_id)
            .order_by(Experience.created_at.desc())
            .limit(1)
        )
        return self.session.scalar(statement)

    # Topic question banks

    def get_question_bank(self, topic_id: str) -> dict[str, Any] | None:
        """Return the cached MCQ items for a topic (answers included), or None."""
        statement = select(TopicQuestionBank).where(TopicQuestionBank.topic_id == topic_id)
        bank = self.session.scalars(statement).first()
        return _bank_to_dict(bank) if bank is not None else None

    def save_question_bank(
        self,
        *,
        topic_id: str,
        items: list[dict[str, Any]],
        model_provider: str | None,
        model_name: str | None,
    ) -> dict[str, Any]:
        """Create the question bank for a topic and return it as a dict."""
        bank = TopicQuestionBank(
            topic_id=topic_id,
            items=items,
            n_items=len(items),
            model_provider=model_provider,
            model_name=model_name,
        )
        self.session.add(bank)
        self.session.flush()
        return _bank_to_dict(bank)

    def _require_learner_and_session(self, student_id: str, session_id: str) -> None:
        if self.session.get(LearnerProfile, student_id) is None:
            raise ReferencedRecordNotFoundError(f"Learner {student_id!r} does not exist.")
        if self.session.get(LearningSession, session_id) is None:
            raise ReferencedRecordNotFoundError(f"Learning session {session_id!r} does not exist.")


def _assessment_to_dict(assessment: Assessment) -> dict[str, Any]:
    return {
        "id": assessment.id,
        "session_id": assessment.session_id,
        "student_id": assessment.student_id,
        "topic_id": assessment.topic_id,
        "type": assessment.type,
        "items": assessment.items,
        "answers": assessment.answers,
        "score": assessment.score,
        "competency": assessment.competency,
        "created_at": assessment.created_at,
    }


def _policy_to_dict(policy: Policy) -> dict[str, Any]:
    return {
        "policy_id": policy.policy_id,
        "name": policy.name,
        "top_k": policy.top_k,
        "threshold": policy.threshold,
        "max_retries": policy.max_retries,
        "route": policy.route,
        "is_default": policy.is_default,
        "description": policy.description,
    }


def _experience_to_dict(experience: Experience) -> dict[str, Any]:
    return {
        "id": experience.id,
        "student_id": experience.student_id,
        "session_id": experience.session_id,
        "topic_id": experience.topic_id,
        "state": experience.state,
        "state_vector": [experience.state.get(key) for key in EXPERIENCE_STATE_KEYS],
        "policy_id": experience.policy_id,
        "qoe_before": experience.qoe_before,
        "qoe_after": experience.qoe_after,
        "sre_before": experience.sre_before,
        "sre_after": experience.sre_after,
        "reward": experience.reward,
        "created_at": experience.created_at,
    }


def _bank_to_dict(bank: TopicQuestionBank) -> dict[str, Any]:
    return {
        "topic_id": bank.topic_id,
        "items": bank.items,
        "n_items": bank.n_items,
        "model_provider": bank.model_provider,
        "model_name": bank.model_name,
    }
