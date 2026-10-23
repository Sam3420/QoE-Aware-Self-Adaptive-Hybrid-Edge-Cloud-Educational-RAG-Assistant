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
    # "escalate" (P5): answer with the local SLM, escalate to the cloud LLM when retrieval confidence is low
    # or a grounding check rejects the local answer (agent/escalation.py).
    route: Literal["local", "cloud", "escalate"]


class SessionContext(BaseModel):
    """Context vector used to find similar past sessions."""

    model_config = ConfigDict(extra="forbid")

    rtt_ms: float = Field(ge=0.0)
    cpu: float = Field(ge=0.0, le=1.0)
    mem: float = Field(ge=0.0, le=1.0)
    competency_num: float = Field(ge=0.0, le=1.0)
    last_qoe: float = Field(ge=0.0, le=1.0)


class TopicReply(BaseModel):
    """Resume value for the select_topic interrupt: {"topic_id": <a catalogue topic id>}."""

    topic_id: str

    @field_validator("topic_id")
    @classmethod
    def _in_catalogue(cls, topic_id: str) -> str:
        from knowledge_prep.catalog import TOPICS, get_topic

        topic_id = topic_id.strip()
        if get_topic(topic_id) is None:
            raise ValueError(f"unknown topic; choose one of {[t.topic_id for t in TOPICS]}")
        return topic_id


class VideoReply(BaseModel):
    """Resume value for the select_video interrupt: exactly one of
    {"item_id": <recommendation item id>}, {"refresh": true} (new recommendations) or {"end": true}."""

    item_id: str | None = None
    refresh: bool = False
    end: bool = False

    @model_validator(mode="after")
    def _exactly_one(self) -> "VideoReply":
        self.item_id = (self.item_id or "").strip() or None
        if sum([self.item_id is not None, self.refresh, self.end]) != 1:
            raise ValueError('send {"item_id": "<id>"}, {"refresh": true} or {"end": true}')
        return self


class QuestionReply(BaseModel):
    """Resume value for the ask_question interrupt: {"query": str} to ask a doubt, {"more_videos": true} to
    choose another video, or {"end": true} to go to the post-test."""

    query: str | None = Field(default=None, max_length=2000)
    end: bool = False
    more_videos: bool = False
    accept_suggestion: bool = False  # {"accept_suggestion": true}: do what the coach's suggestion button offers

    @model_validator(mode="before")
    @classmethod
    def _accept_plain_string(cls, data: Any) -> Any:
        """Allow a bare question string as shorthand for {"query": ...}."""
        return {"query": data} if isinstance(data, str) else data

    @model_validator(mode="after")
    def _query_unless_end(self) -> "QuestionReply":
        """Require a non-blank query unless the learner is ending the session."""
        self.query = (self.query or "").strip() or None
        if self.end and self.more_videos:
            raise ValueError('send either {"more_videos": true} or {"end": true}, not both')
        if self.accept_suggestion and (self.end or self.more_videos):
            raise ValueError('send {"accept_suggestion": true} on its own')
        if not (self.end or self.more_videos or self.accept_suggestion) and self.query is None:
            raise ValueError('send {"query": "<question>"}, {"more_videos": true} or {"end": true}')
        return self


class FormativeReply(BaseModel):
    """Resume value for one formative_check question: {"answer": "A"-"D" or 0-3}."""

    answer: str | int

    @field_validator("answer")
    @classmethod
    def _option(cls, value: str | int) -> str | int:
        from app.services.mcq import normalize_answer

        if normalize_answer(value) is None:
            raise ValueError('answer with one of "A", "B", "C" or "D"')
        return value


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
    resource_id: str | None = None  # index the answer was retrieved from (yt_<video_id>)
    concept_id: str | None = None  # the question's concept (matching, or the accepted suggestion's)
    scaffold: dict[str, str] | None = None  # answer shape (coach_mode rules): style, support, why
    # P5 (route "escalate") only: whether and why the answer was escalated to the cloud LLM.
    escalated: bool = False
    escalation_stage: Literal["before", "after"] | None = None  # low retrieval confidence / failed check
    escalation_reason: str | None = None
    check_supported: bool | None = None  # grounding check verdict on the local answer (None: not checked)
    check_score: float | None = None  # share of the local answer's claims supported by the context
    check_ms: float | None = None  # time spent on the check
    draft_model: str | None = None  # the local model whose answer was replaced
    video_title: str | None = None  # the video being studied when the question was asked
