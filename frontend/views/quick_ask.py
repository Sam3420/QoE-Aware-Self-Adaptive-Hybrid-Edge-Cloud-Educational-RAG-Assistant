"""Quick ask: the FastAPI text-question route (Phase 2 assistant, no retrieval)."""
from typing import Any

import streamlit as st

from app.core.config import get_settings as get_app_settings
from frontend import resources
from frontend.clients import api_client, data_access
from frontend.clients.agent_backend import AgentBackend
from frontend.components import chat
from frontend.components.ui import code_id, page_header
from frontend.utils import state


def _ensure_api_session(backend: AgentBackend, learner_id: str) -> str:
    """Reuse this learner's API chat session, or create a LearningSession row for it."""
    session_id = state.get(state.API_SESSION)
    if session_id and data_access.session_exists(session_id, learner_id):
        return session_id
    session_id = backend.services.app.create_learning_session(learner_id)
    state.put(state.API_SESSION, session_id)
    return session_id


def render(backend: AgentBackend, learner: dict[str, Any] | None) -> None:
    page_header("Quick ask", "FastAPI · POST /sessions/{id}/interactions/text-question",
                f"Sends your question over HTTP to the backend's AssistantService, which calls the Hugging Face "
                f"model `{get_app_settings().llm_model_id}` and stores the interaction. This route has no "
                f"retrieval or personalization; use Learning session for the RAG agent.")
    if learner is None:
        st.info("Select or create a learner in the sidebar first.")
        return

    health = resources.api_health(api_client.DEFAULT_API_URL)
    if not health.ok:
        st.error(f"Backend unavailable: {health.error}")
        st.code("uvicorn app.main:app --host 127.0.0.1 --port 8000", language="bash")
        st.caption("Run it from the repository root so it uses the same .env and data/app.db as this frontend.")

    session_id = state.get(state.API_SESSION)
    if session_id and not data_access.session_exists(session_id, learner["id"]):
        state.put(state.API_SESSION, None)
        session_id = None
    if session_id:
        st.markdown(f"API chat session {code_id(session_id)}", unsafe_allow_html=True)
        rows = data_access.session_interactions(session_id)
        if not rows:
            st.caption("No interactions recorded in this session yet.")
        for row in rows:
            chat.api_interaction(row)
        if st.button("New API chat session", icon=":material/add:"):
            state.put(state.API_SESSION, None)
            st.rerun()
    else:
        st.caption("A learning session is created for you when you send the first question.")

    if err := st.session_state.pop("api_error", None):
        st.error(err["message"], icon=":material/error:")
        if err.get("interaction_id"):
            st.caption(f"The backend recorded this failure as interaction `{err['interaction_id']}` (shown above).")

    if prompt := st.chat_input("Ask the assistant…", max_chars=4000, disabled=not health.ok):
        session_id = _ensure_api_session(backend, learner["id"])
        with st.chat_message("user"):
            st.write(prompt)
        with st.spinner(f"Waiting for the backend and {get_app_settings().llm_model_id}…"):
            try:
                api_client.ask_text_question(session_id, prompt)
            except api_client.ApiError as exc:
                st.session_state["api_error"] = {"message": str(exc), "interaction_id": exc.interaction_id}
        st.rerun()
