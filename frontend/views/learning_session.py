"""Learning session: drives the LangGraph agent through its pre-test, question and post-test interrupts."""
from typing import Any

import pandas as pd
import streamlit as st

from config import settings
from frontend.clients import data_access
from frontend.clients.agent_backend import AgentBackend, StepResult
from frontend.components import chat
from frontend.components.mcq import mcq_inputs
from frontend.components.ui import badge, code_id, page_header, show_step_error
from frontend.utils import state
from frontend.utils.formatting import ms, pct


def _remember(step: StepResult, action: str) -> None:
    """Keep the last call's trace/error so it survives the rerun that follows."""
    st.session_state["last_step"] = {"step": step, "action": action}


def _last_step_panel() -> None:
    last = st.session_state.get("last_step")
    if not last:
        return
    step: StepResult = last["step"]
    if step.error:
        show_step_error(step, last["action"])
    elif step.trace:
        with st.expander(f"Last action: {last['action']} · {len(step.trace)} graph nodes in {ms(step.elapsed_ms)}"):
            st.dataframe(pd.DataFrame(step.trace), hide_index=True, width="stretch")


def _start_panel(backend: AgentBackend, learner: dict[str, Any]) -> None:
    topics = backend.topics()
    if not topics:
        st.warning("No topics found. Add a transcript under knowledge_prep/samples/ or upload one on the "
                   "Knowledge & retrieval page.")
        return

    with st.container(border=True):
        st.subheader("Start a learning session")
        topic = st.selectbox("Topic", list(topics))
        status = backend.index_status(topic)
        if status.exists and status.compatible:
            badge("ok", f"Knowledge index ready · {status.n_chunks} chunks · {status.embedding_model}")
        elif status.exists:
            badge("error", "Index unusable")
            st.caption(status.error)
        else:
            badge("warn", "Knowledge index not found")
            st.caption("Retrieval and MCQ generation need this topic's FAISS index.")
        if status.source and not (status.exists and status.compatible):
            if st.button(f"Build index for '{topic}' now", icon=":material/build:"):
                with st.spinner(f"Chunking and embedding {status.source.name} with {backend.embedder_name}…"):
                    try:
                        meta = backend.build_index(topic, status.source)
                        state.flash("success", f"Indexed {len(meta.chunks)} chunks for '{topic}'.")
                    except Exception as exc:
                        state.flash("error", f"Index build failed: {type(exc).__name__}: {exc}")
                st.rerun()

        providers = backend.provider_statuses()
        if not any(p.available for p in providers):
            st.warning("No language model is reachable (Ollama is down and HF_TOKEN is not set). The session "
                       "will still run, but the pre/post-test cannot be generated and answers will be lesson "
                       "extracts from the model-free fallback, not generated text.", icon=":material/warning:")

        st.markdown(
            "The agent will: load your profile → give a **pre-test** (skipped if you already took one for this "
            "topic) → measure **RTT, CPU and memory** → **select a policy** (P1–P4) from similar past sessions → "
            "answer questions with **CRAG** retrieval and the **local/cloud router** → give a **post-test** and "
            "store the session as an **experience**."
        )
        ready = status.exists and status.compatible
        if st.button("Start session", type="primary", icon=":material/play_arrow:", disabled=not ready):
            with st.spinner("Creating the session and running the graph (pre-test generation can take a while)…"):
                step = backend.start_session(learner["id"], topic)
            _remember(step, "Start session")
            if step.out:
                state.put(state.AGENT_SESSION, step.out["thread_id"])
            st.rerun()


def _policy_panel(values: dict[str, Any]) -> None:
    ctx = values.get("context") or {}
    if not values.get("policy_id"):
        return
    with st.container(border=True):
        cols = st.columns(5)
        cols[0].metric("Policy", values["policy_id"])
        cols[1].metric("Route", values["runtime_config"]["route"])
        cols[2].metric("top_k", values["runtime_config"]["top_k"])
        cols[3].metric("CRAG retries", values["runtime_config"]["max_retries"])
        cols[4].metric("Competency", values.get("competency") or "—")
        st.caption(f"Decision: _{values.get('decision_reason')}_")
        if ctx:
            st.caption(f"Context measured at session start: RTT {ms(ctx['rtt_ms'])} · CPU {pct(ctx['cpu'])} · "
                       f"memory {pct(ctx['mem'])} · last QoE {ctx['last_qoe']}")


def _pretest(backend: AgentBackend, thread_id: str, pause: dict[str, Any]) -> None:
    st.subheader("Pre-test")
    st.caption(f"{pause['prompt']} Generated from the lesson by the LLM and scored by app/services/pretest.py.")
    with st.form("pretest"):
        answers = mcq_inputs(pause["items"], f"pre-{thread_id}")
        if st.form_submit_button("Submit pre-test", type="primary"):
            if None in answers:
                st.error("Answer every question before submitting.")
            else:
                with st.spinner("Scoring, measuring context and selecting a policy…"):
                    _remember(backend.resume(thread_id, {"answers": answers}), "Submit pre-test")
                st.rerun()


def _questions(backend: AgentBackend, thread_id: str, values: dict[str, Any], pause: dict[str, Any]) -> None:
    threshold = values["runtime_config"].get("threshold", settings.crag_retrieval_threshold)
    turns = backend.conversation(thread_id)
    if not turns:
        st.info(f"Ask your first question about **{values['topic_id']}**.")
    for turn in turns:
        chat.agent_turn(turn, threshold)

    left, right = st.columns([3, 1])
    left.caption(f"Question {pause['question_no']} of at most {settings.max_questions_per_session}.")
    if right.button("Finish → post-test", icon=":material/flag:", width="stretch"):
        with st.spinner("Preparing the post-test…"):
            _remember(backend.resume(thread_id, {"end": True}), "Finish session")
        st.rerun()

    if prompt := st.chat_input(f"Ask about {values['topic_id']}…", max_chars=2000):
        with st.chat_message("user"):
            st.write(prompt)
        with st.spinner("Retrieving lesson context (CRAG), routing and generating the answer…"):
            _remember(backend.resume(thread_id, {"query": prompt}), "Ask question")
        st.rerun()


def _posttest(backend: AgentBackend, thread_id: str, values: dict[str, Any], pause: dict[str, Any]) -> None:
    threshold = values["runtime_config"].get("threshold", settings.crag_retrieval_threshold)
    turns = backend.conversation(thread_id)
    if turns:
        with st.expander(f"Conversation ({len(turns)} questions)"):
            for turn in turns:
                chat.agent_turn(turn, threshold)
    st.subheader("Post-test and feedback")
    st.caption(pause["prompt"])
    if not pause["items"]:
        st.warning("This topic has no question bank (pre-test generation failed earlier), so there is no "
                   "post-test. The agent keeps your rating in the session state only: without a question bank, "
                   "app/services/posttest.py cannot store feedback.")
    with st.form("posttest"):
        answers = mcq_inputs(pause["items"], f"post-{thread_id}") if pause["items"] else []
        rating = st.radio("Rate this session", ["No rating", 1, 2, 3, 4, 5], index=0, horizontal=True)
        comment = st.text_area("Comment (optional)", max_chars=2000)
        if st.form_submit_button("Submit and finish", type="primary"):
            if None in answers:
                st.error("Answer every question before submitting.")
            else:
                payload = {"answers": answers, "rating": None if rating == "No rating" else rating,
                           "comment": comment.strip() or None}
                with st.spinner("Scoring, evaluating the session and storing the experience…"):
                    _remember(backend.resume(thread_id, payload), "Submit post-test")
                st.rerun()


def _summary(backend: AgentBackend, thread_id: str, values: dict[str, Any]) -> None:
    st.success("Session finished.", icon=":material/task_alt:")
    pre, post = values.get("pre_score"), values.get("post_score")
    cols = st.columns(4)
    cols[0].metric("Pre-test", pct(pre))
    cols[1].metric("Post-test", pct(post), delta=f"{post - pre:+.0%}" if pre is not None and post is not None else None)
    cols[2].metric("Rating", values.get("rating") or "—")
    cols[3].metric("Questions asked", len(values.get("qa_log", [])))

    ev = values.get("evaluation") or {}
    with st.container(border=True):
        st.markdown("**Session evaluation** (metrics/qoe.py, sre.py, utility.py, reward.py)")
        if ev.get("status") != "completed":
            badge("warn", f"Not evaluated: {ev.get('reason', ev.get('status', 'unknown'))}")
        else:
            cols = st.columns(4)
            cols[0].metric("QoE", f"{ev['qoe_after']:.2f}", delta=f"{ev['delta_qoe']:+.2f}")
            cols[1].metric("Resource efficiency (SRE)", f"{ev['sre_after']:.2f}", delta=f"{ev['delta_sre']:+.2f}")
            cols[2].metric("Reward", f"{ev['reward']:+.3f}")
            predicted = ev.get("predicted_utility")
            cols[3].metric("Utility: observed vs predicted", f"{ev['utility_after']:.2f}",
                           delta=f"{ev['utility_after'] - predicted:+.2f} vs prediction" if predicted is not None
                           else "no prediction (cold start)", delta_color="off")
            if ev.get("baseline_used"):
                st.caption("First evaluated session for this learner: the 'before' scores are the configured baseline.")
            comp = ev.get("qoe_components") or {}
            st.caption("QoE components: " + " · ".join(f"{k} {'—' if v is None else f'{v:.2f}'}" for k, v in comp.items())
                       + f" · evidence coverage {pct(ev.get('qoe_coverage'))}")
            missing = [k for k, v in (ev.get("metrics") or {}).items() if v is None]
            if missing:
                st.caption(f"Unavailable metrics (weights renormalised over the rest): {', '.join(missing)}. "
                           "LQ/SQ/TQ have no data source in this build.")
            for note in ev.get("sre_notes") or []:
                st.caption(f"SRE: {note}")

    experience = data_access.experience_for_session(thread_id)
    if experience:
        st.caption(f"Experience stored in the database (id `{experience['id']}`, policy {experience['policy_id']}, "
                   f"reward {experience['reward']}). Future sessions use it for kNN policy selection.")
    else:
        st.warning("No experience row exists for this session.")

    turns = backend.conversation(thread_id)
    if turns:
        with st.expander(f"Conversation ({len(turns)} questions)"):
            threshold = values["runtime_config"].get("threshold", settings.crag_retrieval_threshold)
            for turn in turns:
                chat.agent_turn(turn, threshold)
    if st.button("Find videos for what I missed", icon=":material/smart_display:"):
        st.session_state["rec_topic"] = values["topic_id"]
        st.session_state["rec_session"] = thread_id
        st.session_state.pop("rec_run", None)
        state.goto("Video recommendations")
        st.rerun()
    if st.button("Start a new session", type="primary", icon=":material/add:"):
        state.put(state.AGENT_SESSION, None)
        st.session_state.pop("last_step", None)
        st.rerun()


def render(backend: AgentBackend, learner: dict[str, Any] | None) -> None:
    page_header("Learning session", "LangGraph agent · agent/runner.py",
                "Pre-test → context → kNN policy → CRAG retrieval → local/cloud LLM router → post-test → "
                "experience memory. Every step runs the real graph and services.")
    if learner is None:
        st.info("Select or create a learner in the sidebar to start.")
        return

    thread_id = state.get(state.AGENT_SESSION)
    snap = backend.get_state(thread_id) if thread_id else None
    if thread_id and snap is None:
        st.warning(f"Session {thread_id} has no agent checkpoint (it may be an API-only session or was deleted).")
        state.put(state.AGENT_SESSION, None)
        thread_id = None
    if snap and snap["values"]["student_id"] != learner["id"]:
        state.put(state.AGENT_SESSION, None)
        snap, thread_id = None, None

    _last_step_panel()
    if not snap:
        _start_panel(backend, learner)
        return

    values, pause = snap["values"], snap["interrupt"]
    st.markdown(f"Topic **{values['topic_id']}** · session {code_id(thread_id)}", unsafe_allow_html=True)
    if pause and pause.get("error"):
        st.warning(f"The agent rejected the last reply and asked again: {pause['error']}")
    if values.get("session_errors"):
        with st.container(border=True):
            st.markdown("**Non-fatal session errors reported by the agent**")
            for err in values["session_errors"]:
                st.markdown(f"- {err}")

    _policy_panel(values)
    if pause and pause["type"] == "pre_test":
        _pretest(backend, thread_id, pause)
    elif pause and pause["type"] == "ask_question":
        _questions(backend, thread_id, values, pause)
    elif pause and pause["type"] == "post_test_and_feedback":
        _posttest(backend, thread_id, values, pause)
    elif snap["finished"]:
        _summary(backend, thread_id, values)
    else:
        st.error(f"Unexpected agent state: next={snap['next']}, interrupt={pause}")
