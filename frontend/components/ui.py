"""Shared presentation helpers: page headers, status badges, error panels."""
from html import escape
from pathlib import Path

import streamlit as st

from frontend.clients.agent_backend import StepResult

_CSS = Path(__file__).resolve().parents[1] / "styles" / "theme.css"

# Status is never shown by color alone: every badge has an icon and a word.
_BADGES = {
    "ok": ("green", ":material/check_circle:"),
    "warn": ("orange", ":material/warning:"),
    "error": ("red", ":material/error:"),
    "off": ("gray", ":material/do_not_disturb_on:"),
    "stub": ("violet", ":material/construction:"),
}


def load_css() -> None:
    st.markdown(f"<style>{_CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_header(title: str, kicker: str, description: str) -> None:
    st.markdown(f"<div class='qoe-page-kicker'>{escape(kicker)}</div>", unsafe_allow_html=True)
    st.title(title)
    st.caption(description)


def badge(state: str, label: str) -> None:
    color, icon = _BADGES[state]
    st.badge(label, color=color, icon=icon)


def code_id(value: str | None) -> str:
    """Inline HTML for an id (use with unsafe_allow_html)."""
    return f"<span class='qoe-id'>{escape(value or '—')}</span>"


def show_step_error(step: StepResult, action: str) -> None:
    """Show a failed agent call exactly as it failed."""
    st.error(f"{action} failed: {step.error}")
    if step.trace:
        st.caption("Nodes that ran before the failure: " + " → ".join(t["node"] for t in step.trace))
    if step.error_traceback:
        with st.expander("Traceback"):
            st.code(step.error_traceback)
