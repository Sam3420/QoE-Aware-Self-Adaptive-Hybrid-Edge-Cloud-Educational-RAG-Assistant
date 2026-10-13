"""Streamlit demo of the current implementation: drive a learning session through the agent graph.

Uses the same runner API as the scripts (agent/runner.py) and shows what happens inside: the nodes
visited, the context vector, the kNN policy decision, CRAG retrieval and retries, the LLM route,
the pre/post-test, the QoE/SRE/reward evaluation and the stored experience.

Modes:
  Offline demo: fake retrieval/LLMs/monitor/MCQs behind the real graph, CRAG, router and policy
                selection, on a private in-memory DB and checkpointer (nothing written to data/).
  Real:         FAISS + Ollama embeddings, local/cloud LLM router, psutil + RTT probe, data/app.db.

Run:
  .venv\\Scripts\\streamlit run streamlit_app.py
"""
import sys
import time
import traceback
import types
from typing import Any

import pandas as pd
import streamlit as st

st.set_page_config(page_title="QoE-Aware Educational RAG Assistant", page_icon="🎓", layout="wide")


@st.cache_resource
def faiss_status() -> str | None:
    """Import faiss; if its DLL cannot load, install a stub so the offline mode still runs. Returns the error."""
    try:
        import faiss  # noqa: F401
        return None
    except ImportError as exc:
        error = str(exc)
        stub = types.ModuleType("faiss")
        stub.Index = object  # used in annotations of services/vector_store.py

        def _unavailable(name: str) -> Any:
            raise AttributeError(f"faiss.{name} is unavailable ({error}); use the offline demo mode.")

        stub.__getattr__ = _unavailable
        sys.modules["faiss"] = stub
        return error


FAISS_ERROR = faiss_status()

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agent.deps import make_fake_services, make_real_services, set_services  # noqa: E402
from agent.graph import build_graph, get_checkpointer  # noqa: E402
from agent.policies import POLICIES  # noqa: E402
from agent.runner import SessionRunner  # noqa: E402
from config import settings  # noqa: E402

LETTERS = "ABCD"
INTERRUPT_NODES = {"pre_test", "ask_question", "post_test_and_feedback"}
STUB_NODES: set[str] = set()
# Same context as FakeMonitor + a beginner, so seeded experiences are near neighbours in kNN.
SEED_CTX = {"rtt_ms": 180.0, "cpu": 0.35, "mem": 0.55, "competency_num": 0.0, "last_qoe": 0.5}


# ---- Environment (services + runner), one per mode -------------------------------------------

def node_detail(node: str, update: dict[str, Any]) -> str:
    """One-line summary of a node's state update."""
    if node == "load_profile" and "profile" in update:
        return f"learner '{update['profile'].get('display_name')}'"
    if node == "pre_test" and "pre_score" in update:
        return f"pre_score={update['pre_score']} → competency={update['competency']}"
    if node == "collect_context" and "context" in update:
        return ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in update["context"].items())
    if node == "select_policy":
        return f"{update.get('policy_id')} {update.get('runtime_config')}"
    if node == "ask_question":
        return "end requested" if update.get("continue_session") is False else f"query={update.get('query')!r}"
    if node == "crag_retrieve" and "retrieval_score" in update:
        return f"best score={update['retrieval_score']:.3f}, chunks={len(update['chunks'])}"
    if node == "corrective_retrieve":
        return f"retry {update.get('crag_retries')}: rewritten → {update.get('retrieval_query')!r}"
    if node == "llm_router":
        return str(update.get("model_used", ""))
    if node == "generate_response" and "model_used" in update:
        return f"answered by {update['model_used']}"
    if node == "post_test_and_feedback":
        return f"post_score={update.get('post_score')} rating={update.get('rating')}"
    if node == "evaluate_session" and "reward" in update:
        return f"QoE={update['qoe']:.2f}, SRE={update['sre']:.2f}, reward={update['reward']:+.3f}"
    if node == "store_experience":
        return "experience stored" if not update.get("session_errors") else str(update["session_errors"])
    return ", ".join(sorted(update)) or "-"


def make_env(mode: str) -> dict[str, Any]:
    """Build services, checkpointer and runner for a mode."""
    trace: list[dict[str, Any]] = []
    if mode == "offline":
        services = make_fake_services()
        checkpointer = InMemorySaver()
    else:
        services = make_real_services()
        checkpointer = get_checkpointer()
    services.app.ensure_ready()

    def on_node(node: str, update: dict[str, Any]) -> None:
        trace.append({"step": len(trace) + 1, "node": node, "detail": node_detail(node, update),
                      "updated keys": ", ".join(sorted(update)), "time": time.strftime("%H:%M:%S")})

    set_services(services)
    runner = SessionRunner(build_graph(checkpointer), on_node=on_node)
    return {"mode": mode, "services": services, "runner": runner, "trace": trace,
            "student_id": None, "session": None}


def env() -> dict[str, Any]:
    """Return the current mode's environment and make its services the active ones."""
    e = st.session_state["envs"][st.session_state["mode"]]
    set_services(e["services"])  # deps keeps a process-wide global; re-point it on every rerun
    return e


def run_safely(fn, *args) -> Any:
    """Call the agent and show any exception instead of crashing the page."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - surface every agent failure in the UI
        st.session_state["last_error"] = (f"{type(exc).__name__}: {exc}", traceback.format_exc())
        return None


# ---- Actions --------------------------------------------------------------------------------

def start_session(e: dict[str, Any], topic: str, learner_name: str, new_learner: bool) -> None:
    """Create (or reuse) the learner and run the graph to its first interrupt."""
    app = e["services"].app
    if e["mode"] == "real" and not e["services"].crag.store.exists(topic):
        from knowledge_prep.build_index import build_topic_index

        with st.spinner(f"Building FAISS index for '{topic}'..."):
            build_topic_index(topic, settings.knowledge_samples_dir / f"{topic}.txt")
    if new_learner or not e["student_id"]:
        e["student_id"] = app.create_learner(learner_name or "Demo Learner")
    e["trace"].clear()
    with st.spinner("Running graph until the first interrupt..."):
        out = run_safely(e["runner"].start_session, e["student_id"], topic)
    if out:
        e["session"] = {"thread_id": out["thread_id"], "topic": topic, "out": out, "chat": [],
                        "started": time.time()}


def resume(e: dict[str, Any], payload: Any, spinner: str = "Running graph...") -> None:
    """Answer the pending interrupt and record any newly answered question."""
    s = e["session"]
    runner = e["runner"]
    before = len(runner.get_state(s["thread_id"])["values"]["qa_log"])
    with st.spinner(spinner):
        out = run_safely(runner.resume, s["thread_id"], payload)
    if not out:
        return
    s["out"] = out
    values = runner.get_state(s["thread_id"])["values"]
    for entry in values["qa_log"][before:]:
        s["chat"].append({"log": entry, "response": values["response"], "chunks": values["chunks"]})


def seed_experiences(e: dict[str, Any], policy_id: str, n: int) -> None:
    """Store n past transitions where policy_id reached a high utility (and P1 a low one), via the real
    experience store so they are indexed in FAISS and used by the policy engine."""
    from agent.experience_store import record_synthetic_transition

    services = e["services"]
    for i in range(n):
        for pid, value in [(policy_id, 0.9 - 0.05 * i), ("P1" if policy_id != "P1" else "P3", 0.3)]:
            record_synthetic_transition(services.experience_store(), services.app, policy_id=pid,
                                        runtime_config=POLICIES[pid].runtime_config(), context=SEED_CTX,
                                        qoe=round(value, 3), sre=round(value, 3))


# ---- Rendering helpers ----------------------------------------------------------------------

def graph_dot(e: dict[str, Any]) -> str:
    """DOT for the compiled graph, with visited nodes, taken edges and the paused node highlighted."""
    g = e["runner"].graph.get_graph()
    visited = [t["node"] for t in e["trace"]]
    s = e["session"]
    paused = s["out"]["interrupt"]["type"] if s and "interrupt" in s["out"] else None
    finished = bool(s and "final_state" in s["out"])
    path = (["__start__"] if visited else []) + visited + (["__end__"] if finished else [])
    taken = set(zip(path, path[1:]))
    counts = {n: visited.count(n) for n in set(visited)}

    lines = ['digraph G {', 'rankdir=TB; bgcolor="transparent"; nodesep=0.35; ranksep=0.32;',
             'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 '
             'fillcolor="#f2f0ff" color="#8b84c9"];',
             'edge [color="#9e9e9e" fontname="Helvetica" fontsize=9];']
    for node_id in g.nodes:
        attrs = []
        label = node_id
        if node_id in ("__start__", "__end__"):
            attrs += ['shape=ellipse', 'fillcolor="#e8e8e8"', 'label="%s"' % node_id.strip("_")]
            if node_id in path:
                attrs.append('fillcolor="#c8e6c9"')
        else:
            if node_id in INTERRUPT_NODES:
                attrs += ['penwidth=2', 'color="#b8860b"']
                label += "\\n⏸ interrupt"
            if node_id in STUB_NODES:
                attrs += ['style="rounded,filled,dashed"']
                label += "\\n(stub)"
            if node_id in counts:
                attrs.append('fillcolor="#c8e6c9"')
                label += f"\\n×{counts[node_id]}"
            if node_id == paused:
                attrs += ['fillcolor="#ffe08a"', 'penwidth=3', 'color="#e65100"']
                label += "\\nWAITING FOR LEARNER"
            attrs.append(f'label="{label}"')
        lines.append(f'"{node_id}" [{" ".join(attrs)}];')
    for edge in g.edges:
        attrs = ['style=dashed'] if edge.conditional else []
        if (edge.source, edge.target) in taken:
            attrs += ['color="#2e7d32"', 'penwidth=2.2']
        lines.append(f'"{edge.source}" -> "{edge.target}" [{" ".join(attrs)}];')
    lines.append("}")
    return "\n".join(lines)


def mcq_form(items: list[dict[str, Any]], key: str) -> list[int | None]:
    """Render MCQ radios and return the selected option indices."""
    answers = []
    for item in items:
        st.markdown(f"**Q{item['question_no']}. {item['question']}**  \n"
                    f"<span style='color:gray;font-size:0.85em'>concept: {item.get('concept', '-')}</span>",
                    unsafe_allow_html=True)
        answers.append(st.radio(f"Q{item['question_no']}", options=range(len(item["options"])), index=None,
                                format_func=lambda i, o=item["options"]: f"{LETTERS[i]}. {o[i]}",
                                key=f"{key}-{item['question_no']}", label_visibility="collapsed"))
    return answers


def render_answer(turn: dict[str, Any]) -> None:
    """One question/answer pair with its retrieval and routing details."""
    log = turn["log"]
    with st.chat_message("user"):
        st.write(log["query"])
    with st.chat_message("assistant", avatar="🎓"):
        st.write(turn["response"])
        route_note = "" if log["model_used"].startswith(log["preferred_route"]) else " ⚠️ fallback"
        st.caption(f"model **{log['model_used']}**{route_note} · retrieval score **{log['retrieval_score']:.3f}** · "
                   f"CRAG retries **{log['crag_retries']}** · chunks **{log['n_chunks']}** · "
                   f"latency **{log['latency_ms']:.0f} ms** (generation {log['gen_latency_ms']:.0f} ms)")
        with st.expander("Retrieval details"):
            if log["retrieval_query"] != log["query"]:
                st.markdown(f"Query rewritten by **corrective_retrieve**: `{log['retrieval_query']}`")
            if log["error"]:
                st.warning(log["error"])
            if turn["chunks"]:
                st.dataframe(pd.DataFrame(turn["chunks"])[["chunk_id", "score", "text"]],
                             hide_index=True, width="stretch")


# ---- Page -----------------------------------------------------------------------------------

st.session_state.setdefault("envs", {})
st.session_state.setdefault("mode", "offline")

with st.sidebar:
    st.title("🎓 QoE-Aware RAG Assistant")
    st.caption("Interactive demo of the current agent (Phases 1–8 + policy selection).")

    modes = ["Offline demo (fakes)"] + ([] if FAISS_ERROR else ["Real (FAISS + Ollama/HF)"])
    mode_label = st.radio("Mode", modes,
                          help="Offline uses deterministic fakes behind the real graph, CRAG, router and policy "
                               "selection, on an in-memory DB. Real uses FAISS, Ollama/HF and data/app.db.")
    if FAISS_ERROR:
        st.warning(f"Real mode is unavailable: FAISS failed to load.\n\n`{FAISS_ERROR}`")
    mode = "offline" if mode_label.startswith("Offline") else "real"
    if mode not in st.session_state["envs"]:
        try:
            with st.spinner("Setting up services..."):
                st.session_state["envs"][mode] = make_env(mode)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not set up {mode} mode: {exc}")
            st.stop()
    st.session_state["mode"] = mode
    e = env()

    st.subheader("Session")
    topics = sorted(p.stem for p in settings.knowledge_samples_dir.glob("*.txt")) or ["photosynthesis"]
    topic = st.selectbox("Topic", topics)
    learner_name = st.text_input("Learner name", "Demo Learner")
    new_learner = st.checkbox("New learner for this session", value=True,
                              help="A returning learner who already took this topic's pre-test skips it.")
    if e["student_id"]:
        st.caption(f"Current learner id: `{e['student_id']}`")
    if st.button("▶ Start new session", type="primary", width="stretch"):
        start_session(e, topic, learner_name, new_learner)
        st.rerun()

    st.subheader("Experience memory")
    n_exps = len(e["services"].app.get_experiences())
    st.caption(f"{n_exps} stored experience(s). kNN needs ≥ {settings.min_experiences}; "
               f"below that the default policy {settings.default_policy_id} is used (cold start).")
    if mode == "offline":
        seed_pid = st.selectbox("Seed past sessions where this policy did best", list(POLICIES), index=1)
        if st.button("Seed 3 good + 3 poor experiences", width="stretch"):
            seed_experiences(e, seed_pid, 3)
            st.rerun()

    with st.expander("Key settings"):
        st.json({"crag_retrieval_threshold": settings.crag_retrieval_threshold,
                 "crag_min_chunk_score": settings.crag_min_chunk_score,
                 "knn_k": settings.knn_k, "min_experiences": settings.min_experiences,
                 "epsilon": settings.epsilon, "max_questions_per_session": settings.max_questions_per_session,
                 "local_llm_model": settings.local_llm_model, "cloud_llm_model": settings.cloud_llm_model,
                 "embedding_backend": settings.embedding_backend})

if err := st.session_state.pop("last_error", None):
    st.error(err[0])
    with st.expander("Traceback"):
        st.code(err[1])

s = e["session"]
values: dict[str, Any] = e["runner"].get_state(s["thread_id"])["values"] if s else {}

tab_session, tab_policy, tab_trace, tab_memory = st.tabs(
    ["🎓 Learner session", "🧠 Context & policy", "🧾 Trace & Q&A log", "🗄️ Experience memory"])

# ---- Tab: learner session ---------------------------------------------------------------------
with tab_session:
    left, right = st.columns([3, 2], gap="large")

    with right:
        st.markdown("##### Agent graph (live)")
        st.caption("🟩 visited · 🟨 waiting for learner · bold green = edges taken · dashed = conditional")
        st.graphviz_chart(graph_dot(e))

    with left:
        if not s:
            st.header("Start a learning session")
            st.markdown(
                "Choose a topic in the sidebar and press **Start new session**. The agent will:\n"
                "1. load the learner profile and give a **pre-test** to estimate competency,\n"
                "2. **collect context** (RTT, CPU, memory, competency, last QoE),\n"
                "3. **select a policy** (P1–P4) from similar past sessions using kNN, or the default on a cold start,\n"
                "4. answer questions with **CRAG** retrieval (rewriting the query when retrieval is weak) "
                "and the **local/cloud LLM router**,\n"
                "5. give a **post-test** and collect feedback, **evaluate** the session and **store the experience** "
                "for future policy decisions.")
            st.info("Tip: in offline mode, seed experiences first to see kNN choose a policy instead of the "
                    "cold-start default. Short questions (≤ 3 words) get weak retrieval and trigger CRAG retries "
                    "under policies that allow them (P2, P4).")
        else:
            out = s["out"]
            st.markdown(f"**Topic:** {s['topic']} · **Session:** `{s['thread_id']}` · "
                        f"**Policy:** {values.get('policy_id') or '—'} · "
                        f"**Competency:** {values.get('competency') or '—'}")
            pause = out.get("interrupt")
            if pause and pause.get("error"):
                st.warning(f"The agent rejected the last reply and asked again: {pause['error']}")

            if pause and pause["type"] == "pre_test":
                st.subheader("📝 Pre-test")
                st.caption(pause["prompt"])
                if e["mode"] == "offline":
                    st.caption("Offline MCQs: the correct answers are A, B, C, D, A.")
                with st.form("pretest"):
                    answers = mcq_form(pause["items"], f"pre-{s['thread_id']}")
                    if st.form_submit_button("Submit pre-test", type="primary"):
                        if None in answers:
                            st.error("Answer every question first.")
                        else:
                            resume(e, {"answers": answers}, "Scoring pre-test, collecting context, selecting policy...")
                            st.rerun()

            elif pause and pause["type"] == "ask_question" or "final_state" in out:
                if values.get("decision_reason"):
                    st.info(f"**Policy {values['policy_id']}** "
                            f"({POLICIES[values['policy_id']].name if values['policy_id'] in POLICIES else ''}) · "
                            f"{values['runtime_config']}  \n_{values['decision_reason']}_")
                for turn in s["chat"]:
                    render_answer(turn)

                if pause:
                    st.markdown(f"**Question {pause['question_no']}** of max {settings.max_questions_per_session}")
                    examples = ["photosynthesis?", "How do plants convert sunlight into chemical energy?",
                                "What is the role of chlorophyll in photosynthesis?"]
                    cols = st.columns(len(examples))
                    for col, ex in zip(cols, examples):
                        if col.button(ex, key=f"ex-{ex}-{pause['question_no']}", width="stretch"):
                            resume(e, {"query": ex}, "Retrieving (CRAG), routing and generating...")
                            st.rerun()
                    with st.form("ask", clear_on_submit=True):
                        q = st.text_input("Your question", placeholder="Ask anything about the topic...")
                        c1, c2 = st.columns(2)
                        ask = c1.form_submit_button("Ask", type="primary", width="stretch")
                        end = c2.form_submit_button("Finish → post-test", width="stretch")
                    if ask:
                        resume(e, {"query": q}, "Retrieving (CRAG), routing and generating...")
                        st.rerun()
                    if end:
                        resume(e, {"end": True}, "Preparing post-test...")
                        st.rerun()

            elif pause and pause["type"] == "post_test_and_feedback":
                for turn in s["chat"]:
                    render_answer(turn)
                st.subheader("📝 Post-test & feedback")
                st.caption(pause["prompt"])
                with st.form("posttest"):
                    answers = mcq_form(pause["items"], f"post-{s['thread_id']}") if pause["items"] else []
                    rating = st.slider("Rate this session", 1, 5, 4)
                    comment = st.text_area("Comment (optional)")
                    if st.form_submit_button("Submit and finish session", type="primary"):
                        if None in answers:
                            st.error("Answer every question first.")
                        else:
                            resume(e, {"answers": answers, "rating": rating, "comment": comment or None},
                                   "Scoring, evaluating and storing the experience...")
                            st.rerun()

            if "final_state" in out:
                f = out["final_state"]
                st.success("Session complete. The experience was stored and will inform future policy selection.")
                ev = f.get("evaluation") or {}
                m = st.columns(5)
                m[0].metric("Pre-test", f"{f['pre_score']:.0%}" if f.get("pre_score") is not None else "—")
                m[1].metric("Post-test", f"{f['post_score']:.0%}" if f.get("post_score") is not None else "—",
                            delta=(f"{f['post_score'] - f['pre_score']:+.0%}"
                                   if f.get("post_score") is not None and f.get("pre_score") is not None else None))
                m[2].metric("Rating", f.get("rating") or "—")
                m[3].metric("QoE", ev.get("qoe_after", "—"), delta=(round(ev["qoe_after"] - ev["qoe_before"], 3)
                                                                   if ev else None))
                m[4].metric("Reward", ev.get("reward", "—"))
                st.caption("QoE/SRE/reward are computed by metrics/ from the session's measurements (docs/adaptive_policy.md).")
                if f.get("session_errors"):
                    st.warning("Session errors:\n\n" + "\n".join(f"- {x}" for x in f["session_errors"]))

            if values.get("session_errors") and "final_state" not in out:
                st.warning("Non-fatal session errors:\n\n" + "\n".join(f"- {x}" for x in values["session_errors"]))

# ---- Tab: context and policy ------------------------------------------------------------------
with tab_policy:
    st.subheader("Context vector (collect_context)")
    ctx = values.get("context") or {}
    if ctx:
        c = st.columns(5)
        c[0].metric("RTT to cloud", f"{ctx['rtt_ms']:.0f} ms")
        c[1].metric("CPU", f"{ctx['cpu']:.0%}")
        c[2].metric("Memory", f"{ctx['mem']:.0%}")
        c[3].metric("Competency (num)", ctx["competency_num"])
        c[4].metric("Last QoE", ctx["last_qoe"])
    else:
        st.caption("Collected after the pre-test.")

    st.subheader("Policy decision (select_policy)")
    if values.get("policy_id"):
        st.markdown(f"Chosen **{values['policy_id']}** → `{values['runtime_config']}`  \n"
                    f"Reason: _{values['decision_reason']}_")
        if values.get("similar_exps"):
            st.markdown("**kNN neighbours** (nearest past sessions in context space)")
            st.dataframe(pd.DataFrame(values["similar_exps"]), hide_index=True, width="stretch")
        else:
            st.caption("No neighbours used (cold start or exploration).")
    else:
        st.caption("Not selected yet.")

    st.subheader("Policy catalogue")
    rows = [{**p.model_dump(), "chosen": "✅" if p.policy_id == values.get("policy_id") else ""}
            for p in POLICIES.values()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Selection: with < {settings.min_experiences} experiences → default {settings.default_policy_id}; "
               f"otherwise with probability ε={settings.epsilon} explore another policy, else pick the policy with "
               f"the best mean reward among the {settings.knn_k} nearest neighbours.")

# ---- Tab: trace and Q&A log ---------------------------------------------------------------------
with tab_trace:
    st.subheader("Node execution trace")
    if e["trace"]:
        st.dataframe(pd.DataFrame(e["trace"]), hide_index=True, width="stretch")
    else:
        st.caption("Start a session to see the nodes run.")
    st.subheader("qa_log")
    if values.get("qa_log"):
        st.dataframe(pd.DataFrame(values["qa_log"]), hide_index=True, width="stretch")
    else:
        st.caption("No questions answered yet.")
    if values:
        with st.expander("Raw agent state"):
            st.json({k: v for k, v in values.items() if k != "test_items"}, expanded=False)

# ---- Tab: experience memory -----------------------------------------------------------------
with tab_memory:
    st.subheader("Stored experiences (newest first)")
    exps = e["services"].app.get_experiences()
    if exps:
        df = pd.DataFrame([{**{k: x[k] for k in ("created_at", "session_id", "topic_id", "policy_id", "reward",
                                                  "qoe_before", "qoe_after", "sre_before", "sre_after")},
                            **(x["state"] or {})} for x in exps])
        current = s["thread_id"] if s else None
        df.insert(0, "this session", df["session_id"].eq(current).map({True: "⭐", False: ""}))
        st.dataframe(df, hide_index=True, width="stretch")
        st.markdown("**Mean reward per policy**")
        # A progress column rather than st.bar_chart: Altair's native deps may be blocked on locked-down Windows.
        summary = df.groupby("policy_id")["reward"].agg(["mean", "count"]).reset_index()
        st.dataframe(summary, hide_index=True, width="stretch", column_config={
            "mean": st.column_config.ProgressColumn("mean reward", min_value=0.0, max_value=1.0, format="%.3f"),
            "count": "sessions"})
    else:
        st.caption("Empty. Finish a session, or seed experiences from the sidebar (offline mode).")
    st.subheader("Policies table (DB)")
    st.dataframe(pd.DataFrame(e["services"].app.list_policies()), hide_index=True, width="stretch")
