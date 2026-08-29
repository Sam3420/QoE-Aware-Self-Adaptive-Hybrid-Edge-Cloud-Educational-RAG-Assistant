from app.db.base import Base
from app.db.models import (
    EducationalResource,
    Interaction,
    LearnerProfile,
    LearningSession,
    RuntimeConfiguration,
)

__all__ = [
    "EducationalResource",
    "Interaction",
    "LearnerProfile",
    "LearningSession",
    "RuntimeConfiguration",
    "create_tables",
]


def create_tables(engine) -> None:
    Base.metadata.create_all(bind=engine)
