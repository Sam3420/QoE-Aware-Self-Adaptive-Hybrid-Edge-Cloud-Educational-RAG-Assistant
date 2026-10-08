"""Learner profile: Phase 3 personalization fields, persisted with LearningTraceRepository."""
from typing import Any

import pandas as pd
import streamlit as st

from app.domain.enums import CompetencyLevel, PreferredLanguage
from frontend.clients import data_access
from frontend.components.ui import badge, code_id, page_header
from frontend.utils import state
from frontend.utils.formatting import pct, when


def _usage_notes() -> None:
    """What each stored field actually changes, per the current source code."""
    with st.container(border=True):
        st.markdown("**How the agent uses these fields**")
        badge("ok", "Preferred language")
        st.caption("Read by `load_profile` and passed to the answer prompt (`build_request`: 'Answer in …').")
        badge("warn", "Competency level")
        st.caption("Stored, but a learning session sets competency from the pre-test score instead "
                   "(beginner if the pre-test is skipped).")
        badge("off", "Interests and learning preferences")
        st.caption("Stored in the database only. No assistant or agent code reads them yet.")


def render(learner: dict[str, Any] | None) -> None:
    page_header("Learner profile", "Database · learner_profiles",
                "Personalization fields saved with LearningTraceRepository.update_learner_personalization.")
    if learner is None:
        st.info("Select or create a learner in the sidebar first.")
        return

    st.markdown(f"**{learner['display_name'] or 'Unnamed learner'}** · {code_id(learner['id'])} · "
                f"created {when(learner['created_at'])}", unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="large")
    with left, st.form("personalization"):
        languages = [None] + [lang.value for lang in PreferredLanguage]
        levels = [None] + [level.value for level in CompetencyLevel]
        prefs = learner["learning_preferences"]
        language = st.selectbox("Preferred language", languages,
                                index=languages.index(learner["preferred_language"]),
                                format_func=lambda v: "Not set (agent uses English)" if v is None else v.title())
        competency = st.selectbox("Competency level", levels, index=levels.index(learner["competency_level"]),
                                  format_func=lambda v: "Not set" if v is None else v.title())
        interests = st.text_input("Interests (comma-separated)", ", ".join(learner["interests"]))
        style = st.text_input("Explanation style", prefs.get("explanation_style") or "",
                              placeholder="e.g. step-by-step, examples first")
        detail = st.text_input("Detail level", prefs.get("detail_level") or "", placeholder="e.g. brief, detailed")
        if st.form_submit_button("Save to database", type="primary"):
            saved = data_access.update_personalization(
                learner["id"], language=language, competency=competency,
                interests=[i.strip() for i in interests.split(",") if i.strip()],
                explanation_style=style.strip(), detail_level=detail.strip())
            if saved is None:
                st.error("Learner no longer exists.")
            else:
                state.flash("success", "Personalization saved to the database.")
                st.rerun()
    with right:
        _usage_notes()

    st.subheader("Assessment history")
    rows = data_access.learner_assessments(learner["id"])
    if not rows:
        st.caption("No pre- or post-tests recorded yet.")
    else:
        df = pd.DataFrame(rows)
        df["created_at"] = df["created_at"].map(when)
        df["score"] = df["score"].map(pct)
        st.dataframe(df, hide_index=True, width="stretch")
