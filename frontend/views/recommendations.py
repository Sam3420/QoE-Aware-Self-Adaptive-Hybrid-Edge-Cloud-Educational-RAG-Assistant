"""Video recommendations: YouTube search + transcript evidence, ranked for the learner, with feedback tracking."""
from typing import Any

import pandas as pd
import streamlit as st

from config import settings
from frontend import resources
from frontend.clients.agent_backend import AgentBackend
from frontend.components.ui import badge, page_header
from frontend.utils import state
from frontend.utils.formatting import ms, pct, when
from knowledge_prep.catalog import topic_label
from services.youtube.ranking import mmss

STATUS_BADGE = {"ok": ("ok", "Ranked"), "degraded": ("warn", "Ranked with reduced evidence"),
                "empty": ("warn", "No suitable videos"), "failed": ("error", "Search failed")}


def _log(recommender: Any, run_id: str, event_type: str, item_id: str, value: dict | None = None) -> None:
    """Widget callback: store one learner event (errors are shown after the rerun, never swallowed)."""
    try:
        recommender.record_event(run_id, event_type, item_id=item_id, value=value)
    except Exception as exc:
        state.flash("error", f"Could not record '{event_type}': {type(exc).__name__}: {exc}")


def _log_rating(recommender: Any, run_id: str, item_id: str, key: str) -> None:
    stars = st.session_state.get(key)
    if stars is not None:  # st.feedback("stars") returns 0-4
        _log(recommender, run_id, "rated", item_id, {"rating": int(stars) + 1})


def _config_panel(recommender: Any) -> bool:
    """Show what is configured; return True when a search can run."""
    client = recommender.pipeline.data_client
    configured = client.configured
    with st.container(horizontal=True):
        badge("ok" if configured else "error", "YouTube Data API key set" if configured else "YOUTUBE_API_KEY missing")
        badge("ok" if settings.transcripts_enabled else "off",
              "Transcripts on (youtube-transcript-api)" if settings.transcripts_enabled else "Transcripts off")
        badge("ok", f"Quota estimate today: {client.quota.used:,} / {client.quota.daily_units:,} units")
    if not configured:
        st.warning("Set `YOUTUBE_API_KEY` in `.env` (YouTube Data API v3 enabled in Google Cloud) and restart "
                   "Streamlit. See docs/recommendations.md.", icon=":material/key:")
    return configured


def _request_form(backend: AgentBackend, recommender: Any, learner: dict[str, Any], enabled: bool) -> None:
    topics = list(backend.topics())
    preset = st.session_state.pop("rec_topic", None)
    with st.form("recommend"):
        st.subheader("What do you want to learn?")
        if topics:
            topic = st.selectbox("Topic", topics, index=topics.index(preset) if preset in topics else 0,
                                 format_func=topic_label)
        else:
            topic = st.text_input("Topic", value=preset or "", max_chars=200)
        objective = st.text_input("Learning objective (optional)", max_chars=500,
                                  placeholder="Default: the topic plus the concepts you got wrong in your last test")
        concepts = st.text_input("Concepts to look for (optional, comma-separated)", max_chars=300)
        revision = st.checkbox("Revision: allow videos I already finished or dismissed")
        needs = recommender.learner_needs(learner["id"], topic) if topic else {}
        if needs.get("gap_concepts"):
            st.caption(f"From your latest {needs['assessment_type']}-test: you missed "
                       f"**{', '.join(needs['gap_concepts'])}**. These are used when you leave the fields empty.")
        submitted = st.form_submit_button("Find videos", type="primary", icon=":material/search:", disabled=not enabled)
    if submitted and topic:
        concept_list = [c.strip() for c in concepts.split(",") if c.strip()] or None
        with st.spinner("Searching YouTube, reading transcripts of the best candidates and ranking…"):
            try:
                run = recommender.recommend(learner["id"], topic, objective=objective or None, concepts=concept_list,
                                            session_id=st.session_state.get("rec_session"), revision=revision)
                st.session_state["rec_run"] = run["id"]
                st.session_state.pop("rec_session", None)
            except Exception as exc:
                state.flash("error", f"Recommendation failed: {type(exc).__name__}: {exc}")
        st.rerun()


def _recommendation_card(recommender: Any, run_id: str, rec: dict[str, Any]) -> None:
    item = rec["item_id"]
    with st.container(border=True):
        left, right = st.columns([1, 3])
        if rec.get("thumbnail_url"):
            left.image(rec["thumbnail_url"], alt=f"Thumbnail of {rec['title']}", width="stretch")
        with right:
            st.markdown(f"**{rec['rank']}. {rec['title']}**")
            meta = [rec.get("channel_title") or "Unknown channel",
                    f"{round(rec['duration_s'] / 60)} min" if rec.get("duration_s") else "duration unknown",
                    str(rec["published_at"])[:10] if rec.get("published_at") else None]
            st.caption(" · ".join(m for m in meta if m))
            with st.container(horizontal=True):
                if rec["evidence_level"] == "transcript":
                    badge("ok", f"Transcript evidence · {rec.get('caption_language')} {rec.get('caption_type')}")
                else:
                    badge("warn", f"Metadata only · transcript {rec['transcript_status'].replace('_', ' ')}")
                badge("ok", f"Score {rec['score']:.2f} · evidence coverage {pct(rec['evidence_coverage'])}")
            st.write(rec["explanation"])

        watch_key = f"watch-{item}"
        with st.container(horizontal=True):
            if st.button("Watch here", key=f"btn-{watch_key}", icon=":material/play_circle:",
                         on_click=_log, args=(recommender, run_id, "opened", item)):
                st.session_state[watch_key] = True
            st.link_button("Open on YouTube", rec["url"], icon=":material/open_in_new:",
                           on_click=_log, args=(recommender, run_id, "opened", item))
            st.button("Helpful", key=f"acc-{item}", icon=":material/thumb_up:",
                      on_click=_log, args=(recommender, run_id, "accepted", item))
            st.button("Not relevant", key=f"dis-{item}", icon=":material/thumb_down:",
                      on_click=_log, args=(recommender, run_id, "dismissed", item))
            st.button("I finished it", key=f"done-{item}", icon=":material/task_alt:",
                      on_click=_log, args=(recommender, run_id, "completed", item),
                      help="Self-reported: the app cannot observe watch time.")
            st.feedback("stars", key=f"rate-{item}", on_change=_log_rating,
                        args=(recommender, run_id, item, f"rate-{item}"))

        evidence = rec.get("transcript_evidence")
        if st.session_state.get(watch_key):
            start = int(evidence["excerpts"][0]["start"]) if evidence and evidence.get("excerpts") else 0
            st.video(rec["url"], start_time=start, alt=f"YouTube video: {rec['title']}")
            st.caption("Opening the player is recorded as 'opened'; it does not prove the video was watched.")

        with st.expander("Why this video"):
            breakdown = pd.DataFrame([{"signal": k, "value": v, "used": v is not None}
                                      for k, v in rec["score_breakdown"].items()])
            st.dataframe(breakdown, hide_index=True, width="stretch",
                         alt="Ranking signals for this video and whether each was available",
                         column_config={"value": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0,
                                                                                 format="%.2f")})
            st.caption("Missing signals (empty) are unknown, not bad: they move the score toward a neutral "
                       f"{settings.rec_missing_feature_prior}, weighted by how much evidence is missing.")
            if evidence:
                st.markdown("**Transcript passages that match your objective**")
                for ex in evidence["excerpts"]:
                    st.markdown(f"- `{mmss(ex['start'])}–{mmss(ex['end'])}` (match {ex['similarity']:.2f}) "
                                f"{ex['text']}")
                if evidence["concepts_covered"] or evidence["concepts_missing"]:
                    covered = ", ".join(f"{c} ({mmss(t)})" for c, t in evidence["concepts_covered"].items()) or "none"
                    st.markdown(f"**Concepts found:** {covered}  \n"
                                f"**Not found in transcript:** {', '.join(evidence['concepts_missing']) or 'none'}")
                st.caption(f"{evidence['on_topic_seconds']:.0f} s of on-topic speech across "
                           f"{evidence['n_chunks']} transcript chunks.")


def _outcome_panel(recommender: Any, run: dict[str, Any]) -> None:
    outcome = recommender.evaluate(run["id"])
    with st.expander("Outcomes, QoE signals and pipeline metrics"):
        rows = [{"metric": k, "value": v["value"], "numerator": v["numerator"], "denominator": v["denominator"],
                 "definition": v["definition"], "note": v["note"] or ""} for k, v in outcome["metrics"].items()]
        st.markdown("**Learner outcomes** (empty = not enough evidence yet)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", alt="Recommendation outcome metrics")
        st.markdown("**QoE inputs by component** (normalised 0–1; not combined into an overall QoE)")
        st.dataframe(pd.DataFrame([{"component": comp, "signal": k, "value": v}
                                   for comp, signals in outcome["qoe_signals"].items() for k, v in signals.items()]),
                     hide_index=True, width="stretch", alt="QoE input signals for this run")
        badge("stub", f"Reward: {outcome['reward'] if outcome['reward'] is not None else 'none yet'} (placeholder formula)")
        st.caption("The reward is set only after explicit feedback (helpful / not relevant / stars). It feeds the "
                   "kNN strategy selection for future runs.")
        m = run["metrics"]
        t = m["timings"]
        st.markdown("**Pipeline**")
        st.dataframe(pd.DataFrame([{
            "search": ms(t["search_ms"]), "metadata": ms(t["metadata_ms"]), "transcripts": ms(t["transcript_ms"]),
            "embedding": ms(t["embedding_ms"]), "ranking": ms(t["ranking_ms"]), "total": ms(t["total_ms"]),
            "quota units": m["quota_units"], "api failure rate": m["rates"]["api_failure_rate"],
            "transcript success": m["rates"]["transcript_success_rate"],
            "filtered": m["rates"]["candidate_filtering_rate"], "cache hits": m["rates"]["cache_hit_rate"],
        }]), hide_index=True, width="stretch", alt="Latency and reliability of this recommendation run")
        filtered = run["request"].get("filtered") or []
        if filtered:
            st.markdown("**Candidates removed**")
            st.dataframe(pd.DataFrame(filtered), hide_index=True, width="stretch", alt="Filtered candidate videos")


def _run_view(recommender: Any, run: dict[str, Any]) -> None:
    kind, label = STATUS_BADGE.get(run["status"], ("warn", run["status"]))
    with st.container(border=True):
        with st.container(horizontal=True):
            badge(kind, label)
            badge("ok", f"Strategy {run['strategy_id']} · {run['strategy'].get('name', '')}")
        st.caption(f"Objective: _{run['objective']}_ · search: `{run['query']}` · {when(run['created_at'])}")
        st.caption(f"Strategy decision: _{run['decision_reason']}_")
        for err in run["errors"]:
            st.caption(f":orange[:material/warning:] {err}")

    if run["status"] == "failed":
        st.error("YouTube search failed, so there are no recommendations. The reason is shown above.")
        if st.button("Retry", icon=":material/refresh:"):
            st.session_state.pop("rec_run", None)
            st.rerun()
        return
    if not run["recommendations"]:
        st.info("No video passed the relevance and availability checks. Try a broader objective or tick "
                "'Revision'.", icon=":material/search_off:")
    shown = st.session_state.setdefault("rec_displayed", set())
    for rec in run["recommendations"]:
        if rec["item_id"] not in shown:  # log each item's first display once
            _log(recommender, run["id"], "displayed", rec["item_id"])
            shown.add(rec["item_id"])
        _recommendation_card(recommender, run["id"], rec)
    _outcome_panel(recommender, run)


def render(backend: AgentBackend, learner: dict[str, Any] | None) -> None:
    page_header("Video recommendations", "YouTube Data API v3 · youtube-transcript-api · agent/recommendations.py",
                "Searches YouTube for your topic, reads the transcripts of the most promising videos, and ranks them by "
                "what they actually teach, your level, language and preferred length.")
    if learner is None:
        st.info("Select or create a learner in the sidebar first.")
        return
    try:
        recommender = resources.get_recommender(backend)
    except Exception as exc:
        st.error(f"The recommender could not be initialised: {type(exc).__name__}: {exc}")
        return

    enabled = _config_panel(recommender)
    run_id = st.session_state.get("rec_run")
    run = recommender.get_run(run_id) if run_id else None
    if run and run["student_id"] != learner["id"]:
        run = None
    if run:
        if st.button("New search", icon=":material/add:"):
            st.session_state.pop("rec_run", None)
            st.rerun()
        _run_view(recommender, run)
    else:
        _request_form(backend, recommender, learner, enabled)

    past = recommender.list_runs(student_id=learner["id"], limit=20)
    if past:
        with st.expander(f"Previous recommendation runs ({len(past)})"):
            for r in past:
                cols = st.columns([3, 1])
                cols[0].caption(f"{when(r['created_at'])} · {r['topic_id']} · {r['strategy_id']} · {r['status']} · "
                                f"reward {r['reward'] if r['reward'] is not None else '—'}")
                if cols[1].button("Open", key=f"open-run-{r['id']}"):
                    st.session_state["rec_run"] = r["id"]
                    st.rerun()
