from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.domain.enums import (
    CompetencyLevel,
    InteractionStatus,
    PreferredLanguage,
    SessionStatus,
)


class LearnerProfileCreate(BaseModel):
    display_name: str | None = None
    profile_data: dict[str, Any] = Field(default_factory=dict)


class LearningPreferences(BaseModel):
    explanation_style: str | None = None
    detail_level: str | None = None


class LearnerPersonalizationUpdate(BaseModel):
    preferred_language: PreferredLanguage | None = None
    competency_level: CompetencyLevel | None = None
    interests: list[str] = Field(default_factory=list)
    learning_preferences: LearningPreferences = Field(default_factory=LearningPreferences)


class LearnerPersonalizationProfile(BaseModel):
    learner_id: str
    preferred_language: PreferredLanguage | None = None
    competency_level: CompetencyLevel | None = None
    interests: list[str] = Field(default_factory=list)
    learning_preferences: LearningPreferences = Field(default_factory=LearningPreferences)


class LearningSessionCreate(BaseModel):
    learner_id: str
    status: SessionStatus = SessionStatus.ACTIVE


class EducationalResourceCreate(BaseModel):
    source: str
    external_resource_id: str
    title: str
    url: HttpUrl | None = None
    topic: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeConfigurationCreate(BaseModel):
    name: str
    version: int = Field(ge=1)
    description: str | None = None
    configuration_data: dict[str, Any] = Field(default_factory=dict)


class AssessmentResultCreate(BaseModel):
    topic: str = Field(min_length=1)
    competency_level: CompetencyLevel
    score: float | None = Field(default=None, ge=0, le=100)
    assessment_data: dict[str, Any] = Field(default_factory=dict)


class AssessmentResultResponse(BaseModel):
    id: str
    learner_id: str
    topic: str
    competency_level: CompetencyLevel
    score: float | None = None
    assessment_data: dict[str, Any] = Field(default_factory=dict)


class TopicCompetency(BaseModel):
    topic: str
    competency_level: CompetencyLevel
    score: float | None = None


class PersonalizationContext(BaseModel):
    learner_id: str
    preferred_language: PreferredLanguage | None = None
    competency_level: CompetencyLevel | None = None
    interests: list[str] = Field(default_factory=list)
    learning_preferences: LearningPreferences = Field(default_factory=LearningPreferences)
    topic_competencies: list[TopicCompetency] = Field(default_factory=list)


class InteractionCreate(BaseModel):
    session_id: str
    runtime_configuration_id: str
    educational_resource_id: str | None = None
    user_input: str
    assistant_output: str | None = None
    occurred_at: datetime | None = None
    response_latency_ms: int | None = Field(default=None, ge=0)
    model_provider: str | None = None
    model_name: str | None = None
    status: InteractionStatus = InteractionStatus.COMPLETED
    error_message: str | None = None


class OrmModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class TextQuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    runtime_configuration_id: str | None = None


class TextQuestionResponse(BaseModel):
    interaction_id: str
    session_id: str
    answer: str
    response_latency_ms: int
    model_provider: str
    model_name: str
    runtime_configuration_id: str
