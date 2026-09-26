from typing import Any

from pydantic import BaseModel, Field, StrictInt, field_validator

from app.domain.enums import AssessmentType, CompetencyLevel


class MCQItem(BaseModel):
    """One multiple-choice question, including its correct option (server-side only)."""

    question: str = Field(min_length=1)
    options: list[str] = Field(min_length=4, max_length=4)
    answer_index: int = Field(ge=0, le=3)
    concept: str = Field(min_length=1)

    @field_validator("options")
    @classmethod
    def options_must_be_non_empty(cls, options: list[str]) -> list[str]:
        if any(not option.strip() for option in options):
            raise ValueError("MCQ options must be non-empty strings.")
        if len({option.strip().lower() for option in options}) != len(options):
            raise ValueError("MCQ options must be distinct.")
        return options


class AssessmentCreate(BaseModel):
    session_id: str
    student_id: str
    topic_id: str = Field(min_length=1)
    type: AssessmentType
    items: list[dict[str, Any]] = Field(default_factory=list)
    answers: list[Any] = Field(default_factory=list)
    score: float = Field(ge=0, le=1)
    competency: CompetencyLevel


class ExperienceState(BaseModel):
    """Context stored with an experience; the field order is the kNN vector order."""

    rtt_ms: float
    cpu: float
    mem: float
    competency_num: float
    last_qoe: float | None = None


class ExperienceCreate(BaseModel):
    student_id: str
    session_id: str
    topic_id: str = Field(min_length=1)
    state: ExperienceState
    policy_id: str
    qoe_before: float | None = None
    qoe_after: float | None = None
    sre_before: float | None = None
    sre_after: float | None = None
    reward: float | None = None


class SessionFeedbackCreate(BaseModel):
    session_id: str
    student_id: str
    rating: StrictInt | None = Field(default=None, ge=1, le=5)
    comment: str | None = None
