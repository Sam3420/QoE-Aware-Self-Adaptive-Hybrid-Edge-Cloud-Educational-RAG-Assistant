"""Experience storage: what the agent's store_experience node calls.

Stores values the graph already computed; it does not select policies or compute metrics.
"""

from collections.abc import Mapping
from typing import Any

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.repositories import AdaptationRepository
from app.db.repositories.adaptation import EXPERIENCE_STATE_KEYS
from app.domain.models import ExperienceCreate, ExperienceState
from app.services.mcq import session_scope
from app.services.pretest import ensure_learner_session

logger = get_logger(__name__)

EVALUATION_KEYS: tuple[str, ...] = ("qoe_before", "qoe_after", "sre_before", "sre_after", "reward")
# last_qoe is legitimately absent on a learner's first session, so it is not required.
REQUIRED_CONTEXT_KEYS: tuple[str, ...] = tuple(key for key in EXPERIENCE_STATE_KEYS if key != "last_qoe")


def store_experience(
    student_id: str,
    session_id: str,
    topic_id: str,
    policy_id: str,
    context: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    *,
    db: Session | None = None,
) -> dict[str, Any]:
    """Persist the session's experience once and return it in ``get_experiences()`` shape.

    ``context`` must hold ``rtt_ms``, ``cpu``, ``mem`` and ``competency_num``; ``last_qoe``
    may be missing or None and is stored as None. Other context keys are ignored.
    ``evaluation`` must hold non-None ``qoe_before``, ``qoe_after``, ``sre_before``,
    ``sre_after`` and ``reward``. ``policy_id`` is stored exactly as given.

    Idempotent per session: if the session already has an experience, that one is returned
    and nothing is inserted. Raises ValueError naming any missing context/evaluation fields.
    """
    missing_context = [key for key in REQUIRED_CONTEXT_KEYS if context.get(key) is None]
    if missing_context:
        raise ValueError(f"store_experience: context is missing {', '.join(missing_context)}.")
    missing_evaluation = [key for key in EVALUATION_KEYS if evaluation.get(key) is None]
    if missing_evaluation:
        raise ValueError(f"store_experience: evaluation is missing {', '.join(missing_evaluation)}.")

    data = ExperienceCreate(
        student_id=student_id,
        session_id=session_id,
        topic_id=topic_id,
        state=ExperienceState(**{key: context.get(key) for key in EXPERIENCE_STATE_KEYS}),
        policy_id=policy_id,
        **{key: evaluation[key] for key in EVALUATION_KEYS},
    )

    with session_scope(db) as session:
        ensure_learner_session(session, student_id, session_id)
        repository = AdaptationRepository(session)
        existing = repository.get_experience_for_session(session_id)
        if existing is not None:
            logger.info(
                "experience_already_stored session_id=%s experience_id=%s", session_id, existing["id"]
            )
            return existing

        experience = repository.insert_experience(data)
        logger.info(
            "experience_stored session_id=%s experience_id=%s policy_id=%s reward=%s",
            session_id,
            experience["id"],
            policy_id,
            experience["reward"],
        )
        return experience
