"""Sidebar: live status, learner selection, navigation and the active session."""
from typing import Any

import streamlit as st

from app.core.config import get_settings as get_app_settings
from config import settings
from frontend import resources
from frontend.clients import api_client, data_access
from frontend.clients.agent_backend import AgentBackend
from frontend.components.ui import badge
from frontend.utils import state
from frontend.utils.formatting import ms, short_id

PAGES = {
    "Dashboard": ":material/dashboard:",
    "Learning session": ":material/school:",
    "Quick ask (API)": ":material/chat:",
    "Learner profile": ":material/person:",
    "Knowledge & retrieval": ":material/library_books:",
    "Video recommendations": ":material/smart_display:",
    "Monitoring": ":material/monitor_heart:",
    "Adaptation & memory": ":material/psychology:",
    "History": ":material/history:",
    "System capabilities": ":material/fact_check:",
}


def _status_block(backend: AgentBackend, db: data_access.DatabaseStatus) -> None:
    health = resources.api_health(api_client.DEFAULT_API_URL)
    st.markdown("**System status**")
    if health.ok:
        badge("ok", f"API connected · {ms(health.latency_ms)}")
    else:
        badge("error", "API unavailable")
        st.caption(health.error)
    if db.ok:
        badge("ok", f"Database · {db.counts.get('learners', 0)} learners")
    else:
        badge("error", "Database error")
        st.caption(db.error)
    for p in resources.provider_statuses(backend):
        label = {"local": "Local (edge)", "cloud": "Cloud"}[p.provider]
        badge("ok" if p.available else "off", f"{label}: {p.model}" if p.available else f"{label}: unavailable")
    if st.button("Refresh status", icon=":material/refresh:", width="stretch"):
        resources.refresh_status()
        st.rerun()


def _learner_block(learners: list[dict[str, Any]]) -> None:
    st.markdown("**Learner**")
    current = state.get(state.LEARNER)
    ids = [None] + [l["id"] for l in learners]
    names = {l["id"]: f"{l['display_name'] or 'Unnamed'} · {short_id(l['id'])}" for l in learners}
    if current not in ids:  # restored id no longer in the DB
        if current:
            state.flash("warning", f"Learner {current} no longer exists in the database.")
        state.switch_learner(None)
        current = None
    choice = st.selectbox("Learner", ids, index=ids.index(current), label_visibility="collapsed",
                          format_func=lambda i: "— Select a learner —" if i is None else names[i])
    if choice != current:
        state.switch_learner(choice)
        st.rerun()

    with st.popover("New learner", icon=":material/person_add:", width="stretch"):
        with st.form("create_learner", clear_on_submit=True, border=False):
            name = st.text_input("Display name", max_chars=80)
            if st.form_submit_button("Create learner", type="primary"):
                if not name.strip():
                    st.error("Enter a display name.")
                else:
                    new_id = resources.get_backend().services.app.create_learner(name.strip())
                    state.switch_learner(new_id)
                    state.flash("success", f"Learner '{name.strip()}' created and stored ({new_id}).")
                    st.rerun()


def render(backend: AgentBackend, db: data_access.DatabaseStatus) -> str:
    """Draw the sidebar and return the selected page."""
    with st.sidebar:
        st.markdown("<div class='qoe-brand'>QoE-Aware Educational RAG Assistant</div>"
                    "<div class='qoe-brand-sub'>Self-adaptive hybrid edge-cloud learning agent</div>",
                    unsafe_allow_html=True)
        st.divider()
        page = st.radio("Navigate", list(PAGES), key="nav", label_visibility="collapsed",
                        format_func=lambda p: f"{PAGES[p]} {p}")
        st.divider()
        if db.ok:
            _learner_block(data_access.list_learners())
            if thread := state.get(state.AGENT_SESSION):
                st.caption(f"Active agent session `{short_id(thread)}`")
            if api := state.get(state.API_SESSION):
                st.caption(f"API chat session `{short_id(api)}`")
        st.divider()
        _status_block(backend, db)
        with st.expander("Runtime configuration"):
            st.caption("Read from config.py / app/core/config.py (.env). Restart Streamlit after editing .env.")
            st.json({
                "api_url": api_client.DEFAULT_API_URL,
                "embedding": backend.embedder_name,
                "local_llm": settings.local_llm_model,
                "cloud_llm (agent)": settings.cloud_llm_model,
                "api_llm (FastAPI)": get_app_settings().llm_model_id,
                "crag_threshold": settings.crag_retrieval_threshold,
                "max_questions": settings.max_questions_per_session,
                "knn_k": settings.knn_k,
                "min_experiences": settings.min_experiences,
                "epsilon": settings.epsilon,
            })
    return page
