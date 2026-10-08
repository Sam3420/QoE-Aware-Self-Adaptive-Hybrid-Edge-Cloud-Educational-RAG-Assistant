"""Dashboard: system status and activity, computed from live checks and stored data."""
from typing import Any

import streamlit as st

from frontend import resources
from frontend.clients import api_client, data_access
from frontend.clients.agent_backend import AgentBackend
from frontend.components.ui import badge, page_header
from frontend.utils import state
from frontend.utils.formatting import ms


def _status_row(backend: AgentBackend, db: data_access.DatabaseStatus) -> None:
    health = resources.api_health(api_client.DEFAULT_API_URL)
    providers = {p.provider: p for p in backend.provider_statuses()}
    topics = backend.topics()
    ready = [t for t in topics if backend.index_status(t).compatible]
    cols = st.columns(4)
    with cols[0].container(border=True):
        st.markdown("**FastAPI backend**")
        badge("ok" if health.ok else "error", "Connected" if health.ok else "Unavailable")
        st.caption(f"/health in {ms(health.latency_ms)}" if health.ok else health.error)
    with cols[1].container(border=True):
        st.markdown("**Database**")
        badge("ok" if db.ok else "error", "Reachable" if db.ok else "Error")
        st.caption(db.url if db.ok else db.error)
    with cols[2].container(border=True):
        st.markdown("**LLM routes**")
        for name in ("local", "cloud"):
            p = providers[name]
            badge("ok" if p.available else "off", f"{name}: {p.model if p.available else 'unavailable'}")
    with cols[3].container(border=True):
        st.markdown("**Knowledge indexes**")
        badge("ok" if ready else "warn", f"{len(ready)} of {len(topics)} topics ready")
        st.caption(f"Embedder {backend.embedder_name}")


def _activity(backend: AgentBackend, db: data_access.DatabaseStatus, learner: dict[str, Any] | None) -> None:
    st.subheader("Activity")
    if not db.ok:
        st.error("Activity cannot be read while the database is unavailable.")
        return
    stats = data_access.interaction_stats()
    exps = db.counts.get("experiences", 0)
    agent_sessions = backend.agent_session_count()
    if not any([db.counts.get("learners"), stats["total"], exps, agent_sessions]):
        st.info("No interactions recorded yet. Create a learner and start a learning session.")
        return
    c = st.columns(5)
    c[0].metric("Learners", db.counts.get("learners", 0))
    c[1].metric("Agent sessions", agent_sessions)
    c[2].metric("Experiences stored", exps)
    c[3].metric("API interactions", stats["total"],
                delta=f"{stats['failed']} failed" if stats["failed"] else None, delta_color="inverse")
    c[4].metric("API mean latency", ms(stats["avg_latency_ms"]) if stats["completed"] else "—")

    thread_id = state.get(state.AGENT_SESSION)
    if learner and thread_id:
        turns = backend.conversation(thread_id)
        st.markdown("**Current agent session**")
        if not turns:
            st.caption("No questions answered yet in this session.")
        else:
            latencies = [t["log"]["latency_ms"] for t in turns]
            models = sorted({t["log"]["model_used"] for t in turns})
            c = st.columns(3)
            c[0].metric("Questions asked", len(turns))
            c[1].metric("Mean end-to-end latency", ms(sum(latencies) / len(latencies)))
            c[2].metric("Models that answered", ", ".join(models))


def render(backend: AgentBackend, db: data_access.DatabaseStatus, learner: dict[str, Any] | None) -> None:
    page_header("Dashboard", "Overview", "Live status checks and counts from the database and agent checkpoints.")
    _status_row(backend, db)
    _activity(backend, db, learner)
    with st.container(border=True):
        st.markdown("**What this build does**")
        st.markdown(
            "- **Working:** learner/session storage, FastAPI text assistant, LangGraph learning sessions with "
            "MCQ pre/post-tests, FAISS + CRAG retrieval, local/cloud/extractive LLM routing, psutil/RTT monitoring, "
            "experience memory and kNN policy selection.\n"
            "- **Placeholder:** QoE/SRE/reward evaluation (`evaluate_session_stub`).\n"
            "- **Not implemented:** YouTube recommendations, speech-to-text, text-to-speech.\n\n"
            "See **System capabilities** for the evidence behind each item."
        )
