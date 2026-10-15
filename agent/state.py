"""Agent state schema shared by every node in the learning-session graph."""
import operator
import uuid
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict):
    """Full state of one learning session; nodes return partial updates of it."""

    # Identity
    student_id: str
    session_id: str
    topic_id: str

    # Learner profile and assessment
    profile: dict[str, Any]
    competency: str
    test_items: list[dict[str, Any]]  # this session's pre-test questions (with bank item_id and reason)
    pretest_plan: dict[str, Any]  # mode (new/returning/unavailable), weak_concepts, previous session, notes
    pre_score: float | None
    post_score: float | None
    pre_result: dict[str, Any]  # correct, total, competency, answers (letters) of this session's pre-test
    post_review: dict[str, Any]  # answer key after the post-test: items (answer, explanation, your answers), note

    # Learning resource: recommended videos, the chosen one, and the index Q&A retrieves from
    recommendations: dict[str, Any]  # run_id, status, errors, strategy_id, decision_reason, items
    video_action: str  # select_video's decision: "video", "refresh" or "end"
    video: dict[str, Any]  # chosen video (video_id, title, url, item_id, ...) plus index details once prepared
    resource_id: str  # FAISS index answering questions (yt_<video_id>); "" until knowledge is prepared
    knowledge_attempts: list[dict[str, Any]]  # videos tried by prepare_knowledge, with failure reasons
    knowledge_error: str  # why no recommended video could be indexed (shown again by select_video)
    videos_watched: list[dict[str, Any]]  # every video studied this session (video_id, title, url, resource_id)

    # Policy selection
    context: dict[str, Any]  # rtt_ms, cpu, mem, competency_num, last_qoe
    similar_exps: list[dict[str, Any]]  # kNN neighbours: policy_id, reward, distance, session_id
    policy_id: str
    runtime_config: dict[str, Any]  # top_k, threshold, max_retries, route
    decision_reason: str  # why select_policy chose policy_id (cold start / greedy / exploration)
    candidate_utilities: dict[str, Any]  # action → estimated utility/support (prediction, before execution)
    prev_state: dict[str, Any]  # RuntimeState s: the learner's last measured state, or a baseline state
    prev_state_at: str | None  # when s was measured (bounds the recommendation window of s_next)
    resource_start: dict[str, Any]  # network counter at context collection (for SRE bandwidth)

    # Current question (reset by ask_question)
    query: str
    retrieval_query: str  # query actually sent to retrieval (rewritten by corrective_retrieve)
    question_started_at: float  # unix time the question arrived, for end-to-end latency
    q_errors: list[str]  # errors hit while answering the current question
    chunks: list[dict[str, Any]]  # RetrievedChunk dicts
    retrieval_score: float
    crag_retries: int
    model_used: str
    response: str

    # Per-question log; the reducer appends, so nodes return only new entries.
    qa_log: Annotated[list[dict[str, Any]], operator.add]
    continue_session: bool

    # Feedback and evaluation
    rating: int | None
    comment: str
    evaluation: dict[str, Any]  # qoe_before, qoe_after, sre_before, sre_after, reward (store_experience shape)
    qoe: float
    sre: float
    reward: float

    # Non-fatal problems outside a single question (probe failure, pre-test generation, storage).
    session_errors: Annotated[list[str], operator.add]


def initial_state(student_id: str, topic_id: str = "", session_id: str | None = None) -> AgentState:
    """Return a fully populated starting state for a new session."""
    return AgentState(
        student_id=student_id,
        session_id=session_id or uuid.uuid4().hex,
        topic_id=topic_id,
        profile={},
        competency="",
        test_items=[],
        pretest_plan={},
        pre_score=None,
        post_score=None,
        pre_result={},
        post_review={},
        recommendations={},
        video_action="",
        video={},
        resource_id="",
        knowledge_attempts=[],
        knowledge_error="",
        videos_watched=[],
        context={},
        similar_exps=[],
        policy_id="",
        runtime_config={},
        decision_reason="",
        candidate_utilities={},
        prev_state={},
        prev_state_at=None,
        resource_start={},
        query="",
        retrieval_query="",
        question_started_at=0.0,
        q_errors=[],
        chunks=[],
        retrieval_score=0.0,
        crag_retries=0,
        model_used="",
        response="",
        qa_log=[],
        continue_session=True,
        rating=None,
        comment="",
        evaluation={},
        qoe=0.0,
        sre=0.0,
        reward=0.0,
        session_errors=[],
    )
