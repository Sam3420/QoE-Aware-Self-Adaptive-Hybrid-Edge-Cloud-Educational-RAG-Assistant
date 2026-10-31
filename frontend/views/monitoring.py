"""Monitoring: live psutil/RTT readings, LLM route health, and recorded latencies."""
import time
from typing import Any

import pandas as pd
import streamlit as st

from config import settings
from frontend import resources
from frontend.clients import api_client, data_access
from frontend.clients.agent_backend import AgentBackend
from frontend.components.ui import badge, page_header
from frontend.utils import state
from frontend.utils.formatting import ms, pct

MAX_READINGS = 20


def _measure(backend: AgentBackend) -> dict[str, Any]:
    cpu, mem = backend.cpu_mem()
    rtt, rtt_error = backend.probe_rtt()
    return {"time": time.strftime("%H:%M:%S"), "cpu": cpu, "mem": mem, "rtt_ms": rtt, "rtt_error": rtt_error}


def _live(backend: AgentBackend) -> None:
    st.subheader("Device and network")
    st.caption(f"Same measurements `collect_context` feeds into policy selection: psutil CPU/memory and one HTTP "
               f"HEAD to `{settings.probe_url}` (timeout {settings.probe_timeout_s} s).")
    readings: list[dict[str, Any]] = st.session_state.setdefault("readings", [])
    if st.button("Measure now", type="primary", icon=":material/speed:"):
        with st.spinner("Sampling CPU for 0.1 s and probing RTT…"):
            readings.append(_measure(backend))
            del readings[:-MAX_READINGS]
    if not readings:
        st.caption("No measurements taken yet in this browser session.")
        return
    last = readings[-1]
    c = st.columns(3)
    c[0].metric("CPU", pct(last["cpu"]))
    c[1].metric("Memory", pct(last["mem"]))
    c[2].metric("RTT to cloud", ms(last["rtt_ms"]) if last["rtt_ms"] is not None else "probe failed")
    if last["rtt_error"]:
        st.warning(last["rtt_error"])
    if len(readings) > 1:
        df = pd.DataFrame(readings)
        df["cpu"], df["mem"] = df["cpu"].map(pct), df["mem"].map(pct)
        df["rtt_ms"] = df["rtt_ms"].map(lambda v: "failed" if v is None else f"{v:.0f}")
        st.dataframe(df.drop(columns="rtt_error").iloc[::-1], hide_index=True, width="stretch")


def _providers(backend: AgentBackend) -> None:
    st.subheader("Services")
    health = resources.api_health(api_client.DEFAULT_API_URL)
    with st.container(border=True):
        badge("ok" if health.ok else "error", f"FastAPI {health.url}")
        st.caption(f"HTTP {health.status_code}, {ms(health.latency_ms)}" if health.ok else health.error)
        for p in backend.provider_statuses():
            badge("ok" if p.available else "off", f"{p.provider} route · {p.model}")
            st.caption(f"{'Available' if p.available else 'Unavailable'} · checked by: {p.how_checked}")
        badge("ok", "extractive fallback · always available (no model)")


def _recorded(thread_id: str | None, backend: AgentBackend) -> None:
    st.subheader("Recorded latencies")
    stats = data_access.interaction_stats(since=data_access.APP_STARTED)
    if stats["total"] == 0:
        st.caption("FastAPI assistant: no interactions since the app started.")
    else:
        c = st.columns(3)
        c[0].metric("API interactions", stats["total"])
        c[1].metric("Failed", stats["failed"])
        c[2].metric("Mean latency (completed)", ms(stats["avg_latency_ms"]))
    turns = backend.conversation(thread_id) if thread_id else []
    if turns:
        st.markdown("**Current agent session** (qa_log)")
        st.dataframe(pd.DataFrame([{
            "question": t["log"]["question_no"], "model": t["log"]["model_used"],
            "end-to-end": ms(t["log"]["latency_ms"]), "generation": ms(t["log"]["gen_latency_ms"]),
            "retrieval score": round(t["log"]["retrieval_score"], 3), "retries": t["log"]["crag_retries"],
        } for t in turns]), hide_index=True, width="stretch")
    else:
        st.caption("Current agent session: no questions answered yet.")


def render(backend: AgentBackend) -> None:
    page_header("Monitoring", "services/monitoring.py · services/llm_router.py",
                "Real measurements only. Values appear after a measurement or a recorded interaction.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        _live(backend)
    with right:
        _providers(backend)
    _recorded(state.get(state.AGENT_SESSION), backend)
    thread = state.get(state.AGENT_SESSION)
    snap = backend.get_state(thread) if thread else None
    ev = (snap or {}).get("values", {}).get("evaluation") or {}
    with st.container(border=True):
        st.markdown("**QoE and resource efficiency of the active session**")
        if ev.get("status") == "completed":
            st.dataframe(pd.DataFrame([{"QoE": ev["qoe_after"], "SRE": ev["sre_after"], "utility": ev["utility_after"],
                                        "reward": ev["reward"], **(ev.get("qoe_components") or {}),
                                        **{f"{k} eff.": v for k, v in (ev.get("sre_components") or {}).items()}}]),
                         hide_index=True, width="stretch", alt="QoE, SRE, utility and reward of the session")
        else:
            st.caption("QoE and SRE are computed when a learning session finishes (after the post-test).")
