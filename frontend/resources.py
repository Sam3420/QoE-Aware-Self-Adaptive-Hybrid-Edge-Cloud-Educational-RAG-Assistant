"""Process-wide backend objects and short-lived status checks, cached across Streamlit reruns."""
import streamlit as st

from frontend.clients import api_client
from frontend.clients.agent_backend import AgentBackend, ProviderStatus

HEALTH_TTL_S = 15


@st.cache_resource(show_spinner="Connecting to the agent, database and vector store…")
def get_backend() -> AgentBackend:
    """Build the real agent services once per process (not cached if construction raises)."""
    return AgentBackend()


@st.cache_data(ttl=HEALTH_TTL_S, show_spinner=False)
def api_health(base_url: str) -> api_client.HealthResult:
    """GET /health, re-checked at most every HEALTH_TTL_S seconds (Refresh clears it)."""
    return api_client.check_health(base_url)


@st.cache_resource(show_spinner=False)
def get_recommender(_backend: AgentBackend):
    """Video recommender sharing the backend's services (embedder, monitor, DB); built once per process."""
    from agent.recommendations import make_recommender

    return make_recommender(_backend.services)


def provider_statuses(backend: AgentBackend) -> list[ProviderStatus]:
    return backend.provider_statuses()


def refresh_status() -> None:
    api_health.clear()
