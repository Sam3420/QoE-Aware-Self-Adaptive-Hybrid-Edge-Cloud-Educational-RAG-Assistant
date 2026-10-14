"""Knowledge preparation (FAISS indexes) and the CRAG retrieval explorer."""
import re

import pandas as pd
import streamlit as st

from config import settings
from frontend.clients import data_access
from frontend.clients.agent_backend import AgentBackend
from frontend.components.ui import page_header
from frontend.utils import state
from knowledge_prep.catalog import topic_label

_TOPIC_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _index_table(backend: AgentBackend) -> None:
    st.subheader("Topic indexes")
    st.caption(f"Embedder: **{backend.embedder_name}** · indexes in `{settings.vector_index_dir}`")
    topics = backend.topics()
    if not topics:
        st.info("No transcripts or indexes found.")
        return
    rows = []
    for topic, source in topics.items():
        s = backend.index_status(topic)
        bank = data_access.question_bank(topic)
        rows.append({
            "topic": topic_label(topic),
            "index": "ready" if s.exists and s.compatible else ("unusable" if s.exists else "missing"),
            "chunks": s.n_chunks, "dim": s.dim, "embedding model": s.embedding_model,
            "MCQ bank": f"{bank['n_items']} items ({bank['model_provider']})" if bank else "not generated",
            "transcript": source.name if source else "—", "problem": s.error or "",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    buildable = [t for t, src in topics.items() if src is not None]
    missing = [t for t in buildable if not backend.index_status(t).compatible]
    if missing and st.button(f"Build all missing indexes ({len(missing)})", icon=":material/build:"):
        built, failed = [], []
        with st.spinner(f"Embedding {len(missing)} topics with {backend.embedder_name}…"):
            for topic in missing:
                try:
                    backend.build_index(topic, topics[topic])
                    built.append(topic)
                except Exception as exc:
                    failed.append(f"{topic}: {type(exc).__name__}: {exc}")
        if built:
            state.flash("success", f"Built {len(built)} index(es): {', '.join(built)}.")
        for message in failed:
            state.flash("error", f"Index build failed for {message}")
        st.rerun()
    if buildable:
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        topic = c1.selectbox("Build or rebuild the index for", buildable, format_func=topic_label)
        if c2.button("Build index", icon=":material/build:", width="stretch"):
            with st.spinner(f"Chunking and embedding '{topic}' with {backend.embedder_name}…"):
                try:
                    meta = backend.build_index(topic, topics[topic])
                    state.flash("success", f"Indexed {len(meta.chunks)} chunks for '{topic}' "
                                           f"({meta.embedding_model}, dim {meta.dim}).")
                except Exception as exc:
                    state.flash("error", f"Index build failed: {type(exc).__name__}: {exc}")
            st.rerun()


def _upload(backend: AgentBackend) -> None:
    with st.expander("Add a topic from a transcript (.txt)"):
        with st.form("upload", clear_on_submit=True):
            topic = st.text_input("Topic id", placeholder="e.g. water_cycle (lowercase, digits, _ or -)")
            file = st.file_uploader("Transcript", type=["txt"])
            if st.form_submit_button("Save and build index", type="primary"):
                if not _TOPIC_RE.match(topic or ""):
                    st.error("Topic id must be lowercase letters, digits, '_' or '-'.")
                elif file is None:
                    st.error("Choose a .txt transcript.")
                else:
                    try:
                        text = file.getvalue().decode("utf-8")
                    except UnicodeDecodeError:
                        st.error("The transcript must be UTF-8 text.")
                        return
                    if not text.strip():
                        st.error("The transcript is empty.")
                        return
                    folder = backend.upload_dir()
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder / f"{topic}.txt"
                    path.write_text(text, encoding="utf-8")
                    with st.spinner("Building the index…"):
                        try:
                            meta = backend.build_index(topic, path)
                            state.flash("success", f"Saved {path.name} and indexed {len(meta.chunks)} chunks.")
                        except Exception as exc:
                            state.flash("error", f"Saved {path.name}, but the index build failed: "
                                                 f"{type(exc).__name__}: {exc}")
                    st.rerun()


def _explorer(backend: AgentBackend) -> None:
    st.subheader("Retrieval explorer")
    st.caption("Runs the agent's own CRAG retriever (services/crag.py) against a topic index: FAISS cosine "
               f"search, then chunks below {settings.crag_min_chunk_score:.2f} are discarded. The agent accepts "
               f"retrieval when the best score reaches {settings.crag_retrieval_threshold:.2f}.")
    ready = [t for t in backend.topics() if backend.index_status(t).compatible]
    if not ready:
        st.info("Build an index first.")
        return
    with st.form("retrieve"):
        c1, c2 = st.columns([1, 3])
        topic = c1.selectbox("Topic", ready, format_func=topic_label)
        query = c2.text_input("Query", placeholder="e.g. What does chlorophyll do?")
        c3, c4 = st.columns(2)
        top_k = c3.slider("top_k", 1, 10, 5)
        corrective = c4.checkbox("Apply the corrective rewrite (as corrective_retrieve does on a retry)")
        submitted = st.form_submit_button("Retrieve", type="primary")
    if not submitted:
        return
    if not query.strip():
        st.error("Enter a query.")
        return
    try:
        with st.spinner("Embedding the query and searching FAISS…"):
            result = backend.retrieve(topic, query.strip(), top_k, corrective=corrective)
    except Exception as exc:
        st.error(f"Retrieval failed: {type(exc).__name__}: {exc}")
        return
    accepted = result.top_score >= settings.crag_retrieval_threshold
    c = st.columns(3)
    c[0].metric("Best score", f"{result.top_score:.3f}")
    c[1].metric("Chunks kept", len(result.chunks))
    c[2].metric("Discarded (below min score)", result.n_discarded)
    if result.query != query.strip():
        st.caption(f"Rewritten query sent to FAISS: `{result.query}` (chunks re-graded against the original).")
    (st.success if accepted else st.warning)(
        f"CRAG verdict: {'accept' if accepted else 'below threshold'} "
        f"({result.top_score:.3f} {'≥' if accepted else '<'} {settings.crag_retrieval_threshold:.2f})."
        + ("" if accepted else " In a session the agent would retry with a rewritten query if its policy allows."))
    if result.chunks:
        st.dataframe(pd.DataFrame([c.model_dump() for c in result.chunks])[["chunk_id", "score", "source", "text"]],
                     hide_index=True, width="stretch",
                     column_config={"score": st.column_config.NumberColumn(format="%.3f")})


def render(backend: AgentBackend) -> None:
    page_header("Knowledge & retrieval", "knowledge_prep · services/vector_store.py · services/crag.py",
                "Transcript → chunks → embeddings → FAISS index per topic, and the CRAG retrieval the agent runs.")
    _index_table(backend)
    _upload(backend)
    st.divider()
    _explorer(backend)
