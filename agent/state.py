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

    # Coaching (agent/nodes/coaching.py): goals for this session, tracker updates per question, progress on the goals
    session_plan: dict[str, Any]  # status, goals (concept_id, name, reason, mastery, attempts), query (concepts, objective)
    tracker_updates: Annotated[list[dict[str, Any]], operator.add]  # per answered question: concept, status
    progress: dict[str, Any]  # goals (status, mastery now vs at planning), touched/total, suggestion
    tracker_snapshot: dict[str, Any]  # concept id -> {mastery, attempts, correct, covered, asked} after the last answer
    suggestion: dict[str, Any]  # the coach's next step: action, concept_id, label, reason, query, after_question
    suggestion_log: Annotated[list[dict[str, Any]], operator.add]  # each suggestion shown: action, concept, accepted
    video_concepts: dict[str, Any]  # video id -> concept ids it covers (None: unknown), cached for the coach
    video_focus: str  # concept name to search videos for (an accepted another_video suggestion), "" otherwise
    formative: dict[str, Any]  # the running formative check: concept_id, items (with answers), results
    formative_result: dict[str, Any]  # the last finished check, shown back at the questions
    formative_seen: Annotated[list[str], operator.add]  # formative item ids used this session (never repeated)
    formative_history: Annotated[list[dict[str, Any]], operator.add]  # every formative answer: concept_id, correct
    coach_mode: str  # "rules" or "off" for this session (settings.coach_mode when it started)
    question_concept: dict[str, Any]  # the current question's concept: question_no, concept_id, confidence, status
    scaffold: dict[str, Any]  # how the current answer was shaped: style, support, why ({} when coach_mode is off)
    mastery_start: dict[str, Any]  # every concept's state at planning time (reflection: before vs after)
    reflection: dict[str, Any]  # end of session: goals met, mastery change, checks, suggestions, summary text

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
    escalation: dict[str, Any]  # P5: stage/reason when this question was escalated to the cloud LLM

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


def initial_state(student_id: str, topic_id: str = "", session_id: str | None = None,
                  coach_mode: str | None = None) -> AgentState:
    """Return a fully populated starting state for a new session."""
    if coach_mode is None:
        from config import settings

        coach_mode = settings.coach_mode
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
        session_plan={},
        tracker_updates=[],
        progress={},
        tracker_snapshot={},
        suggestion={},
        suggestion_log=[],
        video_concepts={},
        video_focus="",
        formative={},
        formative_result={},
        formative_seen=[],
        formative_history=[],
        coach_mode=coach_mode,
        question_concept={},
        scaffold={},
        mastery_start={},
        reflection={},
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
        escalation={},
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
