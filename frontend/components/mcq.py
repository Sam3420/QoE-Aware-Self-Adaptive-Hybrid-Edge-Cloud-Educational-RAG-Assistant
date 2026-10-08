"""MCQ form for the agent's pre-test and post-test interrupts (scoring happens in app/services)."""
from typing import Any

import streamlit as st

LETTERS = "ABCD"


def mcq_inputs(items: list[dict[str, Any]], key: str) -> list[int | None]:
    """Render one radio group per item and return the chosen option indices (None if unanswered)."""
    answers: list[int | None] = []
    for item in items:
        with st.container(border=True):
            st.markdown(f"**Q{item['question_no']}. {item['question']}**")
            st.caption(f"Concept: {item.get('concept') or '—'}")
            answers.append(st.radio(
                f"Answer to question {item['question_no']}",
                options=range(len(item["options"])),
                index=None,
                format_func=lambda i, opts=item["options"]: f"{LETTERS[i]}. {opts[i]}",
                key=f"{key}-{item['question_no']}",
                label_visibility="collapsed",
            ))
    return answers
