"""Unit tests for conditional edges, policies and pydantic schemas."""
import pytest
from pydantic import ValidationError

from agent.policies import POLICIES, get_policy
from agent.routing import continue_session, crag_ok, more_questions, need_pretest
from agent.schemas import PostTestReply, PreTestReply, QuestionReply, RuntimeConfig
from agent.state import initial_state
from config import settings


def _state(**overrides):
    state = initial_state("s1", "photosynthesis")
    state.update(overrides)
    return state


def test_need_pretest_until_a_pre_score_exists():
    assert need_pretest(_state(pre_score=None)) == "pre_test"
    assert need_pretest(_state(pre_score=0.0)) == "collect_context"


@pytest.mark.parametrize(
    ("score", "retries", "max_retries", "expected"),
    [
        (0.9, 0, 1, "llm_router"),  # good score
        (0.5, 0, 1, "corrective_retrieve"),  # low score, retries left
        (0.5, 1, 1, "llm_router"),  # low score, budget spent
        (0.5, 0, 0, "llm_router"),  # no retries allowed
    ],
)
def test_crag_ok(score, retries, max_retries, expected):
    cfg = {"top_k": 3, "threshold": 0.7, "max_retries": max_retries, "route": "local"}
    assert crag_ok(_state(retrieval_score=score, crag_retries=retries, runtime_config=cfg)) == expected


def test_continue_session_after_ask_question():
    assert continue_session(_state(continue_session=True)) == "crag_retrieve"
    assert continue_session(_state(continue_session=False)) == "post_test_and_feedback"


def test_more_questions_respects_cap():
    assert more_questions(_state(qa_log=[{}])) == "ask_question"
    full_log = [{}] * settings.max_questions_per_session
    assert more_questions(_state(qa_log=full_log)) == "post_test_and_feedback"


def test_policies_are_valid_and_default_is_p1():
    assert set(POLICIES) == {"P1", "P2", "P3", "P4"}
    for policy in POLICIES.values():
        RuntimeConfig.model_validate(policy.runtime_config())
    assert get_policy(None).policy_id == "P1"
    assert get_policy("nope").policy_id == "P1"


def test_question_reply_query_or_end():
    assert QuestionReply.model_validate("  what is light? ").query == "what is light?"
    assert QuestionReply.model_validate({"end": True}).end
    for bad in ({"query": "   "}, {}, {"end": False}):
        with pytest.raises(ValidationError):
            QuestionReply.model_validate(bad)


def test_pre_and_post_test_replies():
    assert PreTestReply(answers=["A", 1]).answers == ["A", 1]
    with pytest.raises(ValidationError):
        PreTestReply.model_validate({"answers": "ABCD"})
    assert PostTestReply(answers=["A"]).rating is None  # rating and comment are optional (B allows None)
    with pytest.raises(ValidationError):
        PostTestReply(rating=6)


def test_runtime_config_rejects_unknown_route():
    with pytest.raises(ValidationError):
        RuntimeConfig(top_k=3, threshold=0.7, max_retries=1, route="gpu")
