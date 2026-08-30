from enum import StrEnum


class SessionStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class InteractionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class PreferredLanguage(StrEnum):
    ENGLISH = "english"
    HINDI = "hindi"


class CompetencyLevel(StrEnum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
