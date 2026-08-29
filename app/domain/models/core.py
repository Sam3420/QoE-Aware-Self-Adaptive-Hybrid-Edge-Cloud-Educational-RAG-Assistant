from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.domain.enums import InteractionStatus, SessionStatus


class LearnerProfileCreate(BaseModel):
    display_name: str | None = None
    profile_data: dict[str, Any] = Field(default_factory=dict)


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
