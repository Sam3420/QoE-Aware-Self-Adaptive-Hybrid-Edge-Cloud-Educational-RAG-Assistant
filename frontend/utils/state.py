"""Streamlit session state, mirrored into the URL so a browser refresh restores the same learner/session.

Only ids are kept here; everything shown is re-read from the database and the agent's checkpoints.
"""
from typing import Any

import streamlit as st

LEARNER = "learner"  # LearnerProfile.id
AGENT_SESSION = "session"  # LearningSession.id == LangGraph thread_id
API_SESSION = "api_session"  # LearningSession.id used for FastAPI text questions
PERSISTED = (LEARNER, AGENT_SESSION, API_SESSION)


def init() -> None:
    """Restore persisted ids from the URL once per browser session."""
    if st.session_state.get("_restored_from_url"):
        return
    for key in PERSISTED:
        value = st.query_params.get(key)
        if value:
            st.session_state[key] = value
    st.session_state["_restored_from_url"] = True


def get(key: str) -> Any:
    return st.session_state.get(key)


def put(key: str, value: str | None) -> None:
    """Set (or clear, with None) an id in session state and in the URL."""
    if value is None:
        st.session_state.pop(key, None)
        if key in st.query_params:
            del st.query_params[key]
    else:
        st.session_state[key] = value
        st.query_params[key] = value


def switch_learner(learner_id: str | None) -> None:
    """Change learner; sessions belong to a learner, so drop the current ones."""
    if learner_id != get(LEARNER):
        put(AGENT_SESSION, None)
        put(API_SESSION, None)
        st.session_state.pop("last_step", None)
    put(LEARNER, learner_id)


def flash(kind: str, message: str) -> None:
    """Queue a message to show after the next rerun (kind: success/info/warning/error)."""
    st.session_state.setdefault("_flash", []).append((kind, message))


def show_flashes() -> None:
    for kind, message in st.session_state.pop("_flash", []):
        getattr(st, kind)(message)


def goto(page: str) -> None:
    """Switch page on the next rerun (the nav widget's value cannot change after it is drawn)."""
    st.session_state["_goto"] = page


def apply_goto() -> None:
    """Call before the nav widget is drawn."""
    if page := st.session_state.pop("_goto", None):
        st.session_state["nav"] = page
