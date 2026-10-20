"""Concept mastery panel: the concept tracker (learner_topic_strength) for one learner and topic."""
from typing import Any

import pandas as pd
import streamlit as st

from frontend.clients import data_access
from frontend.components.ui import badge

MASTERY_HELP = ("mastery = (correct + 1) / (attempts + 2): 0.5 means no evidence yet, one right answer gives 0.67, "
                "one wrong answer 0.33. Recorded from every pre-test and post-test answer.")


def _evidence(row: dict[str, Any]) -> str:
    return f"{row['correct']} / {row['attempts']} correct" if row["attempts"] else "no evidence yet"


def _marks(results: list[bool] | None) -> str:
    return " ".join("✓" if ok else "✗" for ok in results) if results else "—"


def mastery_panel(learner_id: str, topic_id: str, *, session_id: str | None = None) -> None:
    """Mastery per concept; with `session_id`, also this session's pre/post-test answers on each concept."""
    data = data_access.concept_mastery(learner_id, topic_id)
    if data["error"]:
        st.warning(data["error"], icon=":material/warning:")
        return
    if data["vocabulary_status"] is None:
        st.caption("This topic has no concept vocabulary yet, so mastery is not tracked "
                   "(`scripts/extract_concepts.py`).")
        return
    if data["vocabulary_status"] == "approved":
        badge("ok", "Concept list approved")
    else:
        badge("warn", "Concept list is a draft (not yet reviewed)")

    rows = data["rows"]
    results = data_access.session_concept_results(learner_id, topic_id, session_id) if session_id else {}
    table = pd.DataFrame([{
        "Concept": r["concept"] + ("" if r["in_vocabulary"] else " (no longer in the list)"),
        "Mastery": r["mastery"],
        **({"Pre-test": _marks(results.get(r["concept_id"], {}).get("pre-test")),
            "Post-test": _marks(results.get(r["concept_id"], {}).get("post-test"))} if session_id else {}),
        "Evidence": _evidence(r),
        "Covered": r["covered"],
        "What it is": r["description"],
    } for r in rows])
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Mastery": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f", help=MASTERY_HELP),
        "Covered": st.column_config.NumberColumn(help="Times the concept came up in a question or a video "
                                                      "(exposure only; it does not change mastery)."),
    })
    tested = [r for r in rows if r["attempts"]]
    st.caption(f"{len(tested)} of {len(rows)} concepts have evidence. {MASTERY_HELP}")
    if session_id and not results and tested:
        st.caption("No answers from this session could be matched to a concept.")
