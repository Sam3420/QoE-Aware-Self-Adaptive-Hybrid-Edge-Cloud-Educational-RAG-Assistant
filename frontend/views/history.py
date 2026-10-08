"""History: every stored learning session of the learner, with its agent Q&A or API interactions."""
from typing import Any

import pandas as pd
import streamlit as st

from config import settings
from frontend.clients import data_access
from frontend.clients.agent_backend import AgentBackend
from frontend.components import chat
from frontend.components.ui import page_header
from frontend.utils import state
from frontend.utils.formatting import pct, short_id, when


def _session_kind(backend: AgentBackend, session: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    snap = backend.get_state(session["id"])
    if snap is not None:
        return ("agent (finished)" if snap["finished"] else "agent (in progress)"), snap
    return ("API chat" if session["api_interactions"] else "empty"), None


def render(backend: AgentBackend, learner: dict[str, Any] | None) -> None:
    page_header("History", "learning_sessions · interactions · LangGraph checkpoints",
                "Sessions stored for the selected learner. Agent sessions are read back from data/checkpoints.db; "
                "API chats from the interactions table.")
    if learner is None:
        st.info("Select a learner in the sidebar first.")
        return
    sessions = data_access.learner_sessions(learner["id"])
    if not sessions:
        st.caption("No sessions recorded for this learner yet.")
        return

    kinds = {s["id"]: _session_kind(backend, s) for s in sessions}
    rows = []
    for s in sessions:
        kind, snap = kinds[s["id"]]
        values = snap["values"] if snap else {}
        rows.append({"session": short_id(s["id"]), "started": when(s["started_at"]), "type": kind,
                     "topic": values.get("topic_id", "—"), "policy": values.get("policy_id") or "—",
                     "questions": len(values.get("qa_log", [])) if snap else s["api_interactions"],
                     "pre": pct(values.get("pre_score")), "post": pct(values.get("post_score"))})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    selected = st.selectbox("Open session", [s["id"] for s in sessions],
                            format_func=lambda i: f"{short_id(i)} · {kinds[i][0]}")
    kind, snap = kinds[selected]
    if snap:
        if not snap["finished"] and st.button("Continue this session", type="primary", icon=":material/play_arrow:"):
            state.put(state.AGENT_SESSION, selected)
            st.session_state.pop("last_step", None)
            state.goto("Learning session")
            st.rerun()
        threshold = (snap["values"].get("runtime_config") or {}).get("threshold", settings.crag_retrieval_threshold)
        turns = backend.conversation(selected)
        if not turns:
            st.caption("No questions were answered in this session.")
        for turn in turns:
            chat.agent_turn(turn, threshold)
        feedback = data_access.session_feedback(selected)
        if feedback:
            st.caption("Stored feedback: " + "; ".join(f"rating {f['rating']}, {f['comment'] or 'no comment'}"
                                                       for f in feedback))
    else:
        interactions = data_access.session_interactions(selected)
        if not interactions:
            st.caption("This session has no interactions.")
        for row in interactions:
            chat.api_interaction(row)
