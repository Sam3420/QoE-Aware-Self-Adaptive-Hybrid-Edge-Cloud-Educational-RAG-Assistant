"""Pydantic models validating the data that enters or is written into AgentState.

AgentState itself stays a TypedDict (LangGraph merges partial dict updates); nodes validate
values through these models and store the result with .model_dump().
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RuntimeConfig(BaseModel):
    """Runtime configuration produced by the selected policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    top_k: int = Field(ge=1, le=20)
    threshold: float = Field(ge=0.0, le=1.0)
    max_retries: int = Field(ge=0, le=5)
    route: Literal["local", "cloud"]


class SessionContext(BaseModel):
    """Context vector used to find similar past sessions."""

    model_config = ConfigDict(extra="forbid")

    rtt_ms: float = Field(ge=0.0)
    cpu: float = Field(ge=0.0, le=1.0)
    mem: float = Field(ge=0.0, le=1.0)
    competency_num: float = Field(ge=0.0, le=1.0)
    last_qoe: float = Field(ge=0.0, le=1.0)


class QuestionReply(BaseModel):
    """Resume value for the ask_question interrupt: {"query": str} to ask, or {"end": true} to finish."""

    query: str | None = Field(default=None, max_length=2000)
    end: bool = False

    @model_validator(mode="before")
    @classmethod
    def _accept_plain_string(cls, data: Any) -> Any:
        """Allow a bare question string as shorthand for {"query": ...}."""
        return {"query": data} if isinstance(data, str) else data

    @model_validator(mode="after")
    def _query_unless_end(self) -> "QuestionReply":
        """Require a non-blank query unless the learner is ending the session."""
        self.query = (self.query or "").strip() or None
        if not self.end and self.query is None:
            raise ValueError('send {"query": "<question>"} or {"end": true}')
        return self


class PreTestReply(BaseModel):
    """Resume value for the pre_test interrupt: one answer per item, letters "A"-"D" or indices 0-3."""

    answers: list[str | int]


class PostTestReply(BaseModel):
    """Resume value for the post_test_and_feedback interrupt."""

    answers: list[str | int] = Field(default_factory=list)
    rating: int | None = Field(default=None, ge=1, le=5)
    comment: str | None = Field(default=None, max_length=2000)


class QALogEntry(BaseModel):
    """One answered question, as recorded in qa_log."""

    question_no: int = Field(ge=1)
    query: str
    retrieval_query: str  # last query sent to retrieval (differs after a corrective rewrite)
    retrieval_score: float = Field(ge=0.0, le=1.0)
    crag_retries: int = Field(ge=0)
    n_chunks: int = Field(ge=0)
    preferred_route: str  # the policy's route
    model_used: str  # provider:model that actually answered
    latency_ms: float = Field(ge=0.0)  # question received → answer ready
    gen_latency_ms: float = Field(ge=0.0)  # LLM call only
    error: str | None = None  # retrieval errors, route fallbacks, generation failures
