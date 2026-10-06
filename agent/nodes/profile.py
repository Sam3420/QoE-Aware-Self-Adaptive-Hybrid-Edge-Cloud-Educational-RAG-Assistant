"""Learner profile loading."""
from typing import Any

from agent.deps import get_services
from agent.state import AgentState


def load_profile(state: AgentState) -> dict[str, Any]:
    """Load the learner profile and, if the topic was already pre-tested, its score and competency."""
    app = get_services().app
    update: dict[str, Any] = {"profile": app.get_learner_profile(state["student_id"])}
    pretest = app.get_latest_pretest(state["student_id"], state["topic_id"])
    if pretest is not None:
        update |= {"pre_score": pretest["score"], "competency": pretest["competency"]}
    return update
