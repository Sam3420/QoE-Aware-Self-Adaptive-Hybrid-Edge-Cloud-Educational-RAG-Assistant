"""Streamlit frontend for the QoE-Aware Self-Adaptive Hybrid Edge-Cloud Educational RAG Assistant.

Run from the repository root (see frontend/README.md):
    .venv\\Scripts\\streamlit run frontend\\streamlit_app.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# app/core/config.py resolves .env and DATABASE_PATH against the working directory; use the same
# directory as the API server so both read one .env and one data/app.db.
os.chdir(REPO_ROOT)

import streamlit as st  # noqa: E402

st.set_page_config(page_title="QoE Educational Assistant", page_icon=":material/school:", layout="wide")

from frontend import resources  # noqa: E402
from frontend.clients import data_access  # noqa: E402
from frontend.components import sidebar  # noqa: E402
from frontend.components.ui import load_css  # noqa: E402
from frontend.utils import state  # noqa: E402
from frontend.views import (  # noqa: E402
    adaptation,
    capabilities,
    dashboard,
    history,
    knowledge,
    learning_session,
    monitoring,
    profile,
    quick_ask,
)


def main() -> None:
    load_css()
    state.init()
    if "nav" not in st.session_state and st.query_params.get("page") in sidebar.PAGES:
        st.session_state["nav"] = st.query_params["page"]
    state.apply_goto()

    try:
        backend = resources.get_backend()
    except Exception as exc:  # the agent, DB or vector store could not be set up: nothing can run
        st.error(f"The backend services could not be initialised: {type(exc).__name__}: {exc}")
        st.caption("Check that you started Streamlit from the repository root with the project's virtualenv, "
                   "and that data/ is writable.")
        st.stop()

    db = data_access.database_status()
    page = sidebar.render(backend, db)
    st.query_params["page"] = page
    state.show_flashes()
    if not db.ok:
        st.error(f"Database unavailable ({db.url}): {db.error}")
        st.stop()

    learner_id = state.get(state.LEARNER)
    learner = data_access.get_learner(learner_id) if learner_id else None

    if page == "Dashboard":
        dashboard.render(backend, db, learner)
    elif page == "Learning session":
        learning_session.render(backend, learner)
    elif page == "Quick ask (API)":
        quick_ask.render(backend, learner)
    elif page == "Learner profile":
        profile.render(learner)
    elif page == "Knowledge & retrieval":
        knowledge.render(backend)
    elif page == "Monitoring":
        monitoring.render(backend)
    elif page == "Adaptation & memory":
        adaptation.render(backend)
    elif page == "History":
        history.render(backend, learner)
    else:
        capabilities.render()


main()
