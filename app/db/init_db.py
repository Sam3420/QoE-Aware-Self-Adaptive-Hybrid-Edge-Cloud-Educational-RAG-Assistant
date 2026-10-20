from app.db.base import Base
from app.db.models import (
    EducationalResource,
    AssessmentResult,
    Interaction,
    LearnerProfile,
    LearningSession,
    RuntimeConfiguration,
    Assessment,
    Policy,
    Experience,
    TopicQuestionBank,
    SessionFeedback,
)

__all__ = [
    "EducationalResource",
    "AssessmentResult",
    "Interaction",
    "LearnerProfile",
    "LearningSession",
    "RuntimeConfiguration",
    "Assessment",
    "Policy",
    "Experience",
    "TopicQuestionBank",
    "SessionFeedback",
    "create_tables",
]


def create_tables(engine) -> None:
    from app.db.migrations import run_migrations

    run_migrations(engine)  # versioned schema changes first (app/db/migrations.py)
    Base.metadata.create_all(bind=engine)
