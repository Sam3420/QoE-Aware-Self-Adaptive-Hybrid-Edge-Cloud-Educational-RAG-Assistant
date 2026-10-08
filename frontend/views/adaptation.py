"""Adaptation & memory: the policy catalogue, the current decision, and stored experiences."""
import pandas as pd
import streamlit as st

from config import settings
from frontend.clients.agent_backend import AgentBackend
from frontend.components.ui import badge, page_header
from frontend.utils import state
from frontend.utils.formatting import when


def render(backend: AgentBackend) -> None:
    page_header("Adaptation & memory", "agent/policy_selection.py · app/services/experience.py",
                "Case-based policy selection: the session context is compared with stored experiences (kNN) and the "
                "policy with the best mean reward among neighbours is chosen; ε-exploration tries others.")
    app = backend.services.app

    st.subheader("Policy catalogue (policies table)")
    st.dataframe(pd.DataFrame(app.list_policies()), hide_index=True, width="stretch")
    st.caption(f"Fewer than {settings.min_experiences} experiences → default policy (cold start). Otherwise "
               f"k={settings.knn_k} nearest neighbours, ε={settings.epsilon}.")

    thread_id = state.get(state.AGENT_SESSION)
    snap = backend.get_state(thread_id) if thread_id else None
    st.subheader("Current session's decision")
    if snap and snap["values"].get("policy_id"):
        v = snap["values"]
        st.markdown(f"**{v['policy_id']}** → `{v['runtime_config']}`  \n_{v['decision_reason']}_")
        st.markdown("Context vector")
        st.dataframe(pd.DataFrame([v["context"]]), hide_index=True, width="stretch")
        if v["similar_exps"]:
            st.markdown("Nearest stored experiences")
            st.dataframe(pd.DataFrame(v["similar_exps"]), hide_index=True, width="stretch")
        else:
            st.caption("No neighbours were used (cold start).")
    else:
        st.caption("No policy selected yet. A policy is chosen after the pre-test of a learning session.")

    st.subheader("Experience memory (experiences table)")
    exps = app.get_experiences()
    with st.container(border=True):
        badge("stub", "Rewards come from the placeholder evaluation")
        st.caption("Experiences are stored for real, but each reward is computed by `evaluate_session_stub`, not "
                   "by measured QoE. Selection is real; the signal it learns from is a placeholder.")
    if not exps:
        st.caption("No experiences stored yet. Finish a learning session to store the first one.")
        return
    st.caption(f"{len(exps)} stored, newest first.")
    df = pd.DataFrame([{
        "created": when(e["created_at"]), "session": e["session_id"][:8], "topic": e["topic_id"],
        "policy": e["policy_id"], "reward": e["reward"], "qoe_before": e["qoe_before"], "qoe_after": e["qoe_after"],
        **{k: e["state"].get(k) for k in ("rtt_ms", "cpu", "mem", "competency_num", "last_qoe")},
    } for e in exps])
    st.dataframe(df, hide_index=True, width="stretch")
    summary = df.groupby("policy")["reward"].agg(["count", "mean"]).reset_index()
    summary.columns = ["policy", "sessions", "mean reward"]
    st.markdown("**Mean reward per policy**")
    st.dataframe(summary, hide_index=True, width="stretch",
                 column_config={"mean reward": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0,
                                                                               format="%.3f")})
