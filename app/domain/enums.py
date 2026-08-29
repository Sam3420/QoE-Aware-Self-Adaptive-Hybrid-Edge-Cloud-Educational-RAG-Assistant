from enum import StrEnum


class SessionStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class InteractionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
