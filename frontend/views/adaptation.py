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
    elif snap is None:
        st.caption("No learning session is open. A policy is chosen when a session's learner picks their first video.")
    elif snap["finished"]:
        st.caption("This session ended before a video was chosen, so no policy was selected and no experience "
                   "was stored.")
    else:
        stage = (snap["interrupt"] or {}).get("type", "starting")
        st.caption(f"No policy selected yet: the session is at '{stage}'. The policy is chosen when the learner "
                   "picks their first video (after the topic, pre-test and video recommendations).")

    usable = backend.services.experience_store().count()
    if usable < settings.min_experiences:
        st.info(f"**Cold start:** {usable} of {settings.min_experiences} usable experiences, so every session uses the "
                "default policy. An experience is stored only when a session has at least one question and a "
                "submitted post-test.", icon=":material/ac_unit:")

    st.subheader("Experience memory (experiences table)")
    exps = app.get_experiences()
    with st.container(border=True):
        badge("ok", "Rewards from measured QoE and SRE (R = μ1·ΔQoE + μ2·ΔSRE)")
        st.caption("Selection uses only experiences with a full transition record (experience_transitions) indexed in "
                   "FAISS. Rows stored before this module existed have placeholder rewards and are not used.")
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
