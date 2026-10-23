"""Rendering of answered questions, from the agent's qa_log and from persisted API interactions."""
from typing import Any

import pandas as pd
import streamlit as st

from config import settings
from frontend.clients import data_access
from frontend.utils.formatting import ms, when


def agent_turn(turn: dict[str, Any], threshold: float) -> None:
    """One agent question/answer with its real retrieval and routing metadata."""
    log = turn["log"]
    provider = log["model_used"].split(":", 1)[0]
    with st.chat_message("user"):
        st.write(log["query"])
        if log.get("video_title"):
            st.caption(f"About: {log['video_title']}")
    with st.chat_message("assistant", avatar=":material/school:"):
        if turn["response"]:
            st.markdown(turn["response"])
        scaffold = log.get("scaffold") or {}
        if scaffold.get("style") == "step_by_step" or scaffold.get("support") == "hint_first":
            shape = " + ".join(s.replace("_", " ") for s in (scaffold.get("style"), scaffold.get("support"))
                               if s not in ("concise", "direct"))
            st.caption(f":material/stairs: Answer shaped {shape} ({scaffold.get('why')}).")
        else:
            st.warning("The answer text for this question is not in the checkpoint history.")

        if provider == "extractive":
            what = ("a verbatim extract of the best retrieved lesson chunk" if log["n_chunks"]
                    else "its fixed 'not found' reply, because no lesson chunk passed the relevance filter")
            st.warning(f"No language model produced this answer. The agent's model-free fallback (ExtractiveClient) "
                       f"returned {what}. Neither the local nor the cloud route was available, or both failed.",
                       icon=":material/info:")
        elif log.get("escalated"):
            when = ("before answering" if log.get("escalation_stage") == "before"
                    else f"after checking the local answer ({log.get('draft_model')})")
            st.info(f"**Escalated to the cloud model {when}:** {log.get('escalation_reason')}.",
                    icon=":material/trending_up:")
        elif log["preferred_route"] == "escalate":
            if log.get("check_supported"):
                st.caption(f"Policy P5: local answer checked against the video ({log['check_score']:.0%} of its "
                           f"claims supported, check took {ms(log['check_ms'])}).")
        elif provider != log["preferred_route"]:
            st.info(f"The policy preferred the **{log['preferred_route']}** route; the router used "
                    f"**{provider}** instead (see errors below).", icon=":material/alt_route:")

        grounded = log["retrieval_score"] >= threshold
        st.caption(
            f"**{log['model_used']}** · end-to-end {ms(log['latency_ms'])} · generation {ms(log['gen_latency_ms'])} · "
            f"retrieval score {log['retrieval_score']:.3f} "
            f"({'≥' if grounded else '<'} threshold {threshold:.2f}: {'grounded' if grounded else 'low confidence'}) · "
            f"CRAG retries {log['crag_retries']} · chunks {log['n_chunks']}"
        )
        with st.expander("Technical details"):
            st.dataframe(pd.DataFrame([
                ("Question no.", log["question_no"]),
                ("Model used (provider:model)", log["model_used"]),
                ("Policy route (preferred)", log["preferred_route"]),
                ("End-to-end latency", ms(log["latency_ms"])),
                ("Generation latency", ms(log["gen_latency_ms"])),
                ("Retrieval query", log["retrieval_query"]),
                ("Best retrieval score", f"{log['retrieval_score']:.3f}"),
                ("CRAG threshold", f"{threshold:.2f}"),
                ("CRAG min chunk score", f"{settings.crag_min_chunk_score:.2f}"),
                ("Corrective retries", log["crag_retries"]),
                ("Chunks passed to the model", log["n_chunks"]),
                *([("Escalated (P5)", f"{log.get('escalation_stage') or 'no'}"
                                      + (f": {log['escalation_reason']}" if log.get("escalation_reason") else "")),
                   ("Grounding check", "not run" if log.get("check_supported") is None else
                    f"score {log['check_score']:.2f} in {ms(log['check_ms'])}")]
                  if log["preferred_route"] == "escalate" else []),
            ], columns=["field", "value"]).astype(str), hide_index=True, width="stretch")
            if log["retrieval_query"] != log["query"]:
                st.markdown(f"Query rewritten by `corrective_retrieve`: `{log['retrieval_query']}`")
            if log["error"]:
                st.warning(f"Errors while answering: {log['error']}")
            if turn["chunks"]:
                st.markdown("**Retrieved chunks** (cosine similarity, FAISS)")
                st.dataframe(pd.DataFrame(turn["chunks"])[["chunk_id", "score", "source", "text"]],
                             hide_index=True, width="stretch",
                             column_config={"score": st.column_config.NumberColumn(format="%.3f")})
            else:
                st.caption("No chunks passed the relevance filter for this question.")


def api_interaction(row: dict[str, Any]) -> None:
    """One persisted FastAPI interaction, as stored in the interactions table."""
    with st.chat_message("user"):
        st.write(row["user_input"])
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        if row["status"] == "completed":
            st.markdown(row["assistant_output"] or "")
            st.caption(f"**{row['model_provider']}:{row['model_name']}** · {ms(row['response_latency_ms'])} · "
                       f"{when(row['occurred_at'])}")
        else:
            st.error(f"Failed: {row['error_message'] or 'no error message stored'} "
                     f"(after {ms(row['response_latency_ms'])}). Recorded as interaction `{row['id']}`.")
        with st.expander("Technical details"):
            config = data_access.runtime_configuration(row["runtime_configuration_id"])
            st.json({
                "interaction_id": row["id"],
                "status": row["status"],
                "response_latency_ms": row["response_latency_ms"],
                "model_provider": row["model_provider"],
                "model_name": row["model_name"],
                "occurred_at": when(row["occurred_at"]),
                "runtime_configuration": config,
            })
