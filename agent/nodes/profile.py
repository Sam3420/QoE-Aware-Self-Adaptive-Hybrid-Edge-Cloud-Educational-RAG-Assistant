"""Learner profile loading."""
from typing import Any

from agent.deps import get_services
from agent.state import AgentState


def load_profile(state: AgentState) -> dict[str, Any]:
    """Load the learner profile (every session then takes a pre-test planned from the learner's history)."""
    return {"profile": get_services().app.get_learner_profile(state["student_id"])}
