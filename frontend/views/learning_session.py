"""Learning session: drives the LangGraph agent through its pre-test, question and post-test interrupts."""
import queue
import threading
from typing import Any

import pandas as pd
import streamlit as st

from config import settings
from frontend import resources
from frontend.clients import api_client, data_access
from frontend.clients.agent_backend import AgentBackend, StepResult
from frontend.components import chat, speech
from frontend.components.mastery import mastery_panel
from frontend.components.mcq import mcq_inputs
from frontend.components.ui import badge, code_id, page_header, show_step_error
from frontend.utils import state
from frontend.utils.formatting import ms, pct
from knowledge_prep.catalog import topic_label, topic_title
from metrics.weights import GROUP_KEYS


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
    with st.container(border=True):
        st.subheader("Start a learning session")
        providers = backend.provider_statuses()
        if not any(p.available for p in providers):
            st.warning("No language model is reachable (Ollama is down and neither GROQ_API_KEY nor HF_TOKEN is set). The session "
                       "will still run, but the pre/post-test cannot be generated and answers will be lesson "
                       "extracts from the model-free fallback, not generated text.", icon=":material/warning:")
        if not settings.youtube_api_key:
            st.warning("YOUTUBE_API_KEY is not set, so no videos can be recommended and the session cannot reach "
                       "the question stage.", icon=":material/warning:")
        st.markdown(
            "The agent will: load your profile → let you **choose a topic** → give a **pre-test** (planned from your "
            "past results on the topic) → **recommend YouTube videos** for the topic, your level and language → "
            "let you **choose a video** → turn its transcript into a searchable index (once per video) → measure "
            "**RTT, CPU and memory** → **select a policy** (P1–P5) → answer your questions from the video with "
            "**CRAG** retrieval and the **local/cloud router** → give a **post-test** on the same questions and "
            "store the session as an **experience**."
        )
        if st.button("Start session", type="primary", icon=":material/play_arrow:"):
            with st.spinner("Creating the session…"):
                step = backend.start_session(learner["id"])
            _remember(step, "Start session")
            if step.out:
                state.put(state.AGENT_SESSION, step.out["thread_id"])
            st.rerun()


def _select_topic(backend: AgentBackend, thread_id: str, learner: dict[str, Any], pause: dict[str, Any]) -> None:
    st.subheader("Choose a topic")
    options = {t["topic_id"]: t for t in pause["topics"]}
    topic = st.selectbox("Topic", list(options), format_func=lambda t: options[t]["label"])
    taken = [a for a in data_access.learner_assessments(learner["id"]) if a["topic_id"] == topic]
    if taken:
        last = taken[0]
        st.caption(f"You have studied this topic before (last {last['type']}-test {pct(last['score'])}). "
                   "Your pre-test will re-check what you missed and add new questions.")
    else:
        st.caption("First time on this topic: you will start with the standard pre-test.")
    status = backend.index_status(topic)
    if not (status.exists and status.compatible):
        st.caption(f"The topic's lesson notes ({status.source.name if status.source else 'missing'}) will be indexed "
                   "first; they ground the pre-test and post-test questions.")
    if st.button("Continue to the pre-test", type="primary", icon=":material/arrow_forward:"):
        with st.spinner("Preparing the topic and planning your pre-test (writing questions can take a minute)…"):
            _remember(backend.resume(thread_id, {"topic_id": topic}), "Choose topic")
        st.rerun()


def _duration(seconds: int | None) -> str:
    if not seconds:
        return "—"
    if seconds >= 3600:
        return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60}:{seconds % 60:02d}"


def _video_card(backend: AgentBackend, thread_id: str, item: dict[str, Any]) -> None:
    with st.container(border=True):
        left, right = st.columns([1, 3])
        if item.get("thumbnail_url"):
            left.image(item["thumbnail_url"], width="stretch")
        right.markdown(f"**{item['rank']}. [{item['title']}]({item['url']})**")
        captions = f" · captions {item['caption_language']} ({item.get('caption_type')})" \
            if item.get("caption_language") else ""
        right.caption(f"{item.get('channel_title') or '—'} · {_duration(item.get('duration_s'))} · "
                      f"score {item['score']:.2f} · evidence: {item.get('evidence_level')}{captions}")
        if "translated" in str(item.get("caption_type") or ""):
            right.caption(":orange[:material/translate: Captions are YouTube's machine translation; answers from "
                          "this video may be less accurate.]")
        right.caption(item.get("explanation") or "")
        if item.get("covers") is not None:
            right.caption(":material/checklist: Covers: " + (", ".join(item["covers"]) or "none of your concepts"))
        if item.get("watched"):
            right.caption(":green[:material/check_circle: Already watched this session]")
        if item.get("unavailable"):
            right.warning(f"Can't be used for questions: {item['unavailable']}", icon=":material/block:")
        elif right.button("Learn from this video", key=f"vid-{item['item_id']}", type="primary",
                          icon=":material/smart_display:"):
            with st.spinner("Fetching the transcript and building the video's index (first time only), then "
                            "measuring context and selecting a policy…"):
                _remember(backend.resume(thread_id, {"item_id": item["item_id"]}), "Choose video")
            st.rerun()


def _select_video(backend: AgentBackend, thread_id: str, values: dict[str, Any], pause: dict[str, Any]) -> None:
    _pretest_score(values)
    _plan_card(values)
    watched = pause.get("watched") or []
    if watched:
        st.subheader("Watch another video")
        st.caption("Watched so far: " + " · ".join(v.get("title") or v["video_id"] for v in watched))
    else:
        st.subheader("Choose a video to learn from")
    st.caption(f"Recommended for **{topic_title(values['topic_id'])}**, your level "
               f"({values.get('competency') or 'unknown'}) and language ({values['profile'].get('language')}). "
               f"Strategy {pause.get('strategy_id') or '—'}: {pause.get('decision_reason') or '—'}")
    if pause.get("notice"):
        st.warning(pause["notice"], icon=":material/warning:")
    if pause.get("status") not in ("ok", None):
        st.error(f"Recommendation status: {pause['status']}", icon=":material/error:")
    for err in pause.get("errors") or []:
        st.caption(f"Note: {err}")
    if not pause["items"]:
        st.info("No videos were found. Try new recommendations or end the session.")
    # videos the assistant can teach from first; those without a usable transcript are kept, but out of the way
    usable = [i for i in pause["items"] if not i.get("unavailable")]
    unusable = [i for i in pause["items"] if i.get("unavailable")]
    blocked = [i for i in unusable if "blocking transcript requests" in str(i.get("unavailable"))
               or "blocked" in str(i.get("unavailable")).lower()]
    if blocked:
        st.warning("YouTube is temporarily blocking caption requests from this network, so the assistant can't check "
                   "or read these videos' transcripts right now. Requests resume automatically after about 15 minutes "
                   "(then use \"Find other videos\"); switching network, e.g. to a phone hotspot, usually helps at once.",
                   icon=":material/wifi_off:")
    elif pause["items"] and not usable:
        st.warning("None of these videos has a transcript the assistant can use. Try \"Find other videos\".",
                   icon=":material/warning:")
    for item in usable:
        _video_card(backend, thread_id, item)
    if unusable:
        with st.expander(f"{len(unusable)} video(s) can't be used for questions"
                         + (" right now (YouTube block)" if blocked else " (no usable transcript)"),
                         icon=":material/block:"):
            for item in unusable:
                _video_card(backend, thread_id, item)
    c1, c2 = st.columns(2)
    if c1.button("Find other videos", icon=":material/refresh:", width="stretch"):
        with st.spinner("Searching YouTube again…"):
            _remember(backend.resume(thread_id, {"refresh": True}), "New recommendations")
        st.rerun()
    if c2.button("Take the post-test" if watched else "End session",
                 icon=":material/flag:" if watched else ":material/stop:", width="stretch"):
        _remember(backend.resume(thread_id, {"end": True}), "Take the post-test" if watched else "End session")
        st.rerun()
    _general_questions(backend, thread_id, values["student_id"])


def _general_questions(backend: AgentBackend, thread_id: str, learner_id: str) -> None:
    """Questions while choosing a video: the Quick-ask route (FastAPI + Groq, no retrieval), one chat per session.

    Doubts about a video's content are asked after choosing it, where answers come from its transcript.
    """
    st.subheader("Ask a question")
    st.caption("Ask anything about the topic while you choose a video. Answers come from the cloud model without "
               "the lesson videos; after choosing a video, your doubts are answered from that video.")
    key = f"api-session-{thread_id}"
    session_id = st.session_state.get(key)
    if session_id and not data_access.session_exists(session_id, learner_id):
        session_id = st.session_state[key] = None
    for row in data_access.session_interactions(session_id) if session_id else []:
        chat.api_interaction(row)
    if err := st.session_state.pop(f"{key}-error", None):
        st.error(err, icon=":material/error:")

    health = resources.api_health(api_client.DEFAULT_API_URL)
    if not health.ok:
        st.caption(f"Unavailable: {health.error}")
    prompt, spoken = speech.question_box("Ask about the topic…", key=f"general-{thread_id}", max_chars=4000,
                                         disabled=not health.ok, session_id=session_id)
    if prompt:
        if session_id is None:
            session_id = st.session_state[key] = backend.services.app.create_learning_session(learner_id)
        with st.chat_message("user"):
            st.write(prompt)
            speech.spoken_caption(spoken)
        with st.spinner("Thinking…"):
            try:
                api_client.ask_text_question(session_id, prompt)
            except api_client.ApiError as exc:
                st.session_state[f"{key}-error"] = str(exc)
        st.rerun()


def _video_panel(values: dict[str, Any]) -> None:
    video = values.get("video") or {}
    if not values.get("resource_id") or not video:
        return
    with st.expander(f"Video: {video.get('title')}", expanded=True, icon=":material/smart_display:"):
        if video.get("url"):
            st.video(video["url"])
        chunks = f", {video['n_chunks']} chunks" if video.get("n_chunks") else ""
        reused = ", reused from an earlier session" if video.get("reused_index") else ""
        st.caption(f"Answers are retrieved from this video's transcript (index `{values['resource_id']}`"
                   f"{chunks}{reused}).")
        if video.get("replaces"):
            st.info(f"'{video['replaces']['title']}' had no usable transcript, so the next recommendation is used.",
                    icon=":material/swap_horiz:")


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


def _plan_banner(plan: dict[str, Any]) -> None:
    """Explain why the learner got these questions (from the agent's pre-test plan)."""
    if plan.get("mode") == "returning":
        prev = plan.get("previous") or {}
        scores = " → ".join(f"{label} {pct(prev[key])}" for label, key in (("pre-test", "pre_score"),
                                                                               ("post-test", "post_score"))
                            if prev.get(key) is not None)
        # Small models sometimes write a sentence as the concept name; keep the banner readable.
        weak = [c if len(c) <= 40 else c[:37].rstrip() + "…" for c in (plan.get("weak_concepts") or [])[:4]]
        st.info(f"**Welcome back.** Last time on this topic ({str(prev.get('date', ''))[:10]}): {scores or 'no score'}. "
                + (f"This quiz re-checks what you missed and adds new questions on **{', '.join(weak)}**."
                   if weak else "You got everything right last time, so this quiz uses new questions."),
                icon=":material/history:")
    else:
        st.info("**First time on this topic.** This short quiz finds your starting level, so explanations match it.",
                icon=":material/flag:")


def _pretest(backend: AgentBackend, thread_id: str, pause: dict[str, Any]) -> None:
    st.subheader("Pre-test")
    _plan_banner(pause.get("plan") or {})
    st.caption(f"{pause['prompt']} Questions are generated from the lesson by the LLM and scored by "
               "app/services/pretest.py.")
    with st.form("pretest"):
        answers = mcq_inputs(pause["items"], f"pre-{thread_id}")
        if st.form_submit_button("Submit pre-test", type="primary"):
            if None in answers:
                st.error("Answer every question before submitting.")
            else:
                with st.spinner("Scoring and searching YouTube for videos at your level…"):
                    _remember(backend.resume(thread_id, {"answers": answers}), "Submit pre-test")
                st.rerun()


def _pretest_score(values: dict[str, Any]) -> None:
    """The learner's pre-test result, shown once it is scored (answers stay hidden until the post-test)."""
    result = values.get("pre_result") or {}
    if values.get("pre_score") is None or not result:
        return
    with st.container(border=True):
        cols = st.columns([1, 1, 3], vertical_alignment="center")
        cols[0].metric("Pre-test score", f"{result['correct']} / {result['total']}")
        cols[1].metric("Starting level", str(result["competency"]).capitalize())
        cols[2].caption(f"You got {pct(values['pre_score'])} right. The correct answers and explanations are shown "
                        "after the post-test, which asks these same questions again, so you can see what you "
                        "learned in this session.")


_GOAL_STATUS = {"not started": ("gray", ":material/radio_button_unchecked:"),
                "covered by video": ("blue", ":material/smart_display:"),
                "discussed": ("orange", ":material/forum:"), "practised": ("green", ":material/check_circle:")}


def _plan_card(values: dict[str, Any]) -> None:
    """This session's goals (agent plan_session) and, once questions start, progress on each (monitor_progress)."""
    plan = values.get("session_plan") or {}
    goals = plan.get("goals") or []
    if not goals:
        return
    progress = {g["concept_id"]: g for g in (values.get("progress") or {}).get("goals") or []}
    with st.container(border=True):
        st.markdown("**:material/flag: Your goals for this session**")
        for goal in goals:
            now = progress.get(goal["concept_id"])
            name_col, status_col = st.columns([4, 1], vertical_alignment="center")
            name_col.markdown(f"**{goal['name']}** · {goal['reason']}")
            name_col.caption(goal["description"])
            if now:
                color, icon = _GOAL_STATUS.get(now["status"], ("gray", None))
                status_col.badge(now["status"].capitalize(), color=color, icon=icon)
        if values.get("progress", {}).get("suggestion"):
            st.caption(values["progress"]["suggestion"])
        elif plan.get("query", {}).get("concepts"):
            st.caption("The videos below were searched for these goals.")


def _review(values: dict[str, Any]) -> None:
    """Answer key after the post-test: correct option, why it is right, and the learner's pre/post answers."""
    review = values.get("post_review") or {}
    items = review.get("items") or []
    if not items:
        return
    st.subheader("Answers and explanations")
    if review.get("note"):
        st.warning(review["note"], icon=":material/warning:")
    for item in items:
        mark = ":green[:material/check_circle:]" if item["post_correct"] else ":red[:material/cancel:]"
        with st.container(border=True):
            st.markdown(f"{mark} **Q{item['question_no']}. {item['question']}**")
            for i, option in enumerate(item["options"]):
                letter = "ABCD"[i]
                tags = [t for t, hit in (("correct answer", letter == item["answer"]),
                                         ("your post-test answer", letter == item["post_answer"]),
                                         ("your pre-test answer", letter == item.get("pre_answer"))) if hit]
                text = f"{letter}. {option}" + (f"  _({', '.join(tags)})_" if tags else "")
                st.markdown(f":green[**{text}**]" if letter == item["answer"] else
                            (f":red[{text}]" if letter == item["post_answer"] else text))
            if item["explanation"]:
                st.markdown(f"**Why {item['answer']} is right:** {item['explanation']}")
            else:
                st.caption("No explanation is available for this question.")
            progress = ("" if item.get("pre_correct") is None else
                        " · learned this session" if item["post_correct"] and not item["pre_correct"] else
                        " · got it wrong after getting it right in the pre-test" if item["pre_correct"]
                        and not item["post_correct"] else "")
            st.caption(f"Concept: {item['concept']}{progress}")
    st.caption("Explanations are written by the LLM from the lesson text when the question is generated, so check "
               "them against your textbook.")


def _questions(backend: AgentBackend, thread_id: str, values: dict[str, Any], pause: dict[str, Any]) -> None:
    _pretest_score(values)
    _plan_card(values)
    _video_panel(values)
    threshold = values["runtime_config"].get("threshold", settings.crag_retrieval_threshold)
    turns = backend.conversation(thread_id)
    for turn in turns:
        chat.agent_turn(turn, threshold)

    _formative_result(pause.get("formative_result"))
    video_title = (pause.get("video") or {}).get("title") or "the video"
    with st.chat_message("assistant", avatar=":material/school:"):
        if turns:
            st.markdown(f"Any more doubts about **{video_title}**? Ask below. Or would you like to watch another "
                        "video, or are you ready for the post-test?")
        else:
            st.markdown(f"Watch **{video_title}** above, then ask me anything you didn't understand. When you're "
                        "done, you can watch another video or take the post-test.")
        _suggestion_button(backend, thread_id, pause.get("suggestion"))
        c1, c2 = st.columns(2)
        if c1.button("Watch another video", icon=":material/video_library:", width="stretch"):
            with st.spinner("Loading the video recommendations…"):
                _remember(backend.resume(thread_id, {"more_videos": True}), "Watch another video")
            st.rerun()
        if c2.button("Take the post-test", icon=":material/flag:", type="primary", width="stretch"):
            with st.spinner("Preparing the post-test…"):
                _remember(backend.resume(thread_id, {"end": True}), "Take the post-test")
            st.rerun()
    st.caption(f"Question {pause['question_no']} of at most {settings.max_questions_per_session} this session.")

    if turns:  # Accessibility setting: read the newest answer aloud once
        speech.read_aloud_once(turns[-1]["response"], key=speech.answer_key(thread_id, len(turns)))
    prompt, spoken = speech.question_box(f"Ask a doubt about {video_title}…", key=f"ask-{thread_id}")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
            speech.spoken_caption(spoken)
        with st.chat_message("assistant", avatar=":material/school:"):
            with st.spinner("Finding the answer in the video…"):
                step = _stream_answer(backend, thread_id, {"query": prompt})
        _remember(step, "Ask question")
        st.rerun()


_DONE = object()


def _stream_answer(backend: AgentBackend, thread_id: str, payload: dict[str, Any]) -> StepResult:
    """Run the question step in a worker thread and show the answer while the model writes it.

    The streamed text is a preview: after the rerun the stored turn (with sources and checks) replaces it.
    """
    texts: queue.Queue[Any] = queue.Queue()
    result: dict[str, StepResult] = {}

    def work() -> None:
        try:
            result["step"] = backend.resume_streaming(thread_id, payload, texts.put)
        finally:
            texts.put(_DONE)

    worker = threading.Thread(target=work, name="answer-stream", daemon=True)
    worker.start()
    preview, shown = st.empty(), ""
    while (item := texts.get()) is not _DONE:
        shown = "" if item is None else shown + item  # None: this route failed, another one will answer
        preview.markdown(shown + " ▌" if shown else "")
    worker.join()
    return result["step"]


_SUGGESTION_ICONS = {"formative_check": ":material/quiz:", "another_video": ":material/video_library:",
                     "review": ":material/replay:", "post_test": ":material/flag:", "follow_up": ":material/lightbulb:"}


def _suggestion_button(backend: AgentBackend, thread_id: str, suggestion: dict[str, Any] | None) -> None:
    """The coach's one optional next step (agent/coach.py). Ignoring it is fine: the other buttons and the chat
    input work as before, and the agent logs whether it was accepted."""
    if not suggestion:
        return
    action = suggestion["action"]
    if st.button(suggestion["label"], icon=_SUGGESTION_ICONS.get(action), type="primary", width="stretch",
                 key=f"suggestion-{thread_id}"):
        if action in ("review", "follow_up"):  # sends a templated question: stream its answer like a typed one
            with st.spinner("Finding the answer in the video…"):
                step = _stream_answer(backend, thread_id, {"accept_suggestion": True})
        else:
            with st.spinner("One moment…"):
                step = backend.resume(thread_id, {"accept_suggestion": True})
        _remember(step, f"Suggestion: {suggestion['label']}")
        st.rerun()
    st.caption(f":material/tips_and_updates: {suggestion['reason']}")


def _formative_feedback(feedback: dict[str, Any]) -> None:
    if feedback["correct"]:
        st.success(f"Correct: **{feedback['answer']}**. {feedback['explanation']}", icon=":material/check_circle:")
    else:
        st.error(f"Not quite: you chose {feedback['your_answer'] or '—'}, the answer is **{feedback['answer']}**. "
                 f"{feedback['explanation']}", icon=":material/cancel:")


def _formative_result(result: dict[str, Any] | None) -> None:
    """The practice check just finished: score, and feedback on its last question (the earlier ones were shown)."""
    if not result:
        return
    with st.container(border=True):
        if not result.get("total"):
            st.caption(f"Practice check on **{result.get('concept')}** skipped: {result.get('note')}.")
            return
        st.markdown(f"**Practice check on {result['concept']}: {result['correct']} / {result['total']} correct**")
        _formative_feedback(result["results"][-1])


def _formative(backend: AgentBackend, thread_id: str, pause: dict[str, Any]) -> None:
    """One formative question at a time, with instant feedback on the previous answer."""
    st.subheader(f"Quick check: {pause['concept']}")
    st.caption(f"Question {pause['index'] + 1} of {pause['total']}. Practice only: this doesn't change your test "
               "scores, but it updates how well the agent thinks you know this concept.")
    if pause.get("feedback"):
        _formative_feedback(pause["feedback"])
    item = pause["item"]
    speech.listen_button(f"{item['question']} " + " ".join(f"Option {'ABCD'[i]}: {o}." for i, o in
                                                          enumerate(item["options"])), key=f"form-{item['item_id']}")
    with st.form(f"formative-{thread_id}-{item['item_id']}"):
        st.markdown(f"**{item['question']}**")
        choice = st.radio("Your answer", options=range(len(item["options"])), index=None,
                          format_func=lambda i: f"{'ABCD'[i]}. {item['options'][i]}", label_visibility="collapsed")
        if st.form_submit_button("Check my answer", type="primary"):
            if choice is None:
                st.error("Choose an answer first.")
            else:
                with st.spinner("Checking…"):
                    _remember(backend.resume(thread_id, {"answer": "ABCD"[choice]}), "Practice answer")
                st.rerun()


def _posttest(backend: AgentBackend, thread_id: str, values: dict[str, Any], pause: dict[str, Any]) -> None:
    threshold = values["runtime_config"].get("threshold", settings.crag_retrieval_threshold)
    turns = backend.conversation(thread_id)
    if turns:
        with st.expander(f"Conversation ({len(turns)} questions)"):
            for turn in turns:
                chat.agent_turn(turn, threshold)
    _pretest_score(values)
    _plan_card(values)
    st.subheader("Post-test and feedback")
    st.caption(pause["prompt"] + " After you submit, you will see your score, the correct answers and why "
               "they are right.")
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
                with st.spinner("Scoring, preparing the answer explanations, evaluating the session and storing "
                                "the experience…"):
                    _remember(backend.resume(thread_id, payload), "Submit post-test")
                st.rerun()


def _reflection_card(reflection: dict[str, Any]) -> None:
    """End-of-session reflection (agent reflect node): summary, goals met, mastery change per concept."""
    if reflection.get("status") != "ok":
        return
    with st.container(border=True):
        st.markdown("**:material/psychology: Looking back on this session**")
        st.markdown(reflection["summary"])
        if reflection["goals"]:
            for goal in reflection["goals"]:
                icon = ":material/check_circle:" if goal["met"] else ":material/radio_button_unchecked:"
                st.markdown(f"{icon} **{goal['name']}**: {goal['before']:.0%} → {goal['after']:.0%}")
        changed = [c for c in reflection["concepts"] if not any(g["concept_id"] == c["concept_id"]
                                                               for g in reflection["goals"])]
        if changed:
            st.caption("Other concepts: " + " · ".join(f"{c['name']} {c['before']:.0%} → {c['after']:.0%}"
                                                       for c in changed))
        rates = [f"{reflection['formative_checks']} practice check(s)"]
        if reflection.get("suggestions_shown"):
            rates.append(f"{reflection['suggestions_accepted']}/{reflection['suggestions_shown']} suggestions taken")
        if reflection.get("goal_completion_rate") is not None:
            rates.append(f"goal completion {reflection['goal_completion_rate']:.0%}")
        st.caption(" · ".join(rates) + f" · coach mode: {reflection['coach_mode']}")


def _summary(backend: AgentBackend, thread_id: str, values: dict[str, Any]) -> None:
    st.success("Session finished.", icon=":material/task_alt:")
    video = values.get("video") or {}
    if values.get("resource_id") and video:
        st.caption(f"Learned from the video [{video.get('title')}]({video.get('url')}).")
    pre, post = values.get("pre_score"), values.get("post_score")
    pre_n, post_n = values.get("pre_result") or {}, values.get("post_review") or {}

    def score(value: float | None, counts: dict[str, Any]) -> str:
        if value is None:
            return "—"
        return f"{counts['correct']} / {counts['total']} ({pct(value)})" if counts.get("total") else pct(value)

    cols = st.columns(4)
    cols[0].metric("Pre-test", score(pre, pre_n))
    cols[1].metric("Post-test", score(post, post_n),
                   delta=f"{post - pre:+.0%}" if pre is not None and post is not None else None)
    cols[2].metric("Rating", values.get("rating") or "—")
    cols[3].metric("Questions asked", len(values.get("qa_log", [])))
    _review(values)
    _reflection_card(values.get("reflection") or {})
    st.subheader("Concept mastery after this session")
    st.caption("Your pre-test and post-test answers on each concept of the topic, and your mastery across all "
               "sessions so far.")
    mastery_panel(values["student_id"], values["topic_id"], session_id=thread_id)

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
            weighted = {k for keys in GROUP_KEYS.values() for k in keys}  # MG/GC/SA are stored, not weighted
            missing = [k for k, v in (ev.get("metrics") or {}).items() if v is None and k in weighted]
            if missing:
                st.caption(f"Unavailable metrics (weights renormalised over the rest): {', '.join(missing)}. "
                           "SQ/TQ have no data source in this build.")
            lq = ev.get("lq") or {}
            if lq.get("value") is not None:
                st.caption(f"LQ: Langfuse evaluator scores on {lq['scored']} of {lq['answers']} answers.")
            elif lq.get("reason"):
                st.caption(f"LQ unavailable: {lq['reason']}.")
            for note in ev.get("sre_notes") or []:
                st.caption(f"SRE: {note}")

    experience = data_access.experience_for_session(thread_id)
    if experience:
        st.caption(f"Experience stored in the database (id `{experience['id']}`, policy {experience['policy_id']}, "
                   f"reward {experience['reward']}). Future sessions use it for kNN policy selection.")
    else:
        reason = (values.get("evaluation") or {}).get("reason")
        st.warning(f"No experience was stored for this session: {reason}." if reason
                   else "No experience row exists for this session.")

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
    topic_text = topic_label(values["topic_id"]) if values["topic_id"] else "not chosen yet"
    st.markdown(f"Topic **{topic_text}** · session {code_id(thread_id)}", unsafe_allow_html=True)
    if pause and pause.get("error"):
        st.warning(f"The agent rejected the last reply and asked again: {pause['error']}")
    if values.get("session_errors"):
        with st.container(border=True):
            st.markdown("**Non-fatal session errors reported by the agent**")
            for err in values["session_errors"]:
                st.markdown(f"- {err}")

    _policy_panel(values)
    if pause and pause["type"] == "select_topic":
        _select_topic(backend, thread_id, learner, pause)
    elif pause and pause["type"] == "select_video":
        _select_video(backend, thread_id, values, pause)
    elif pause and pause["type"] == "pre_test":
        _pretest(backend, thread_id, pause)
    elif pause and pause["type"] == "ask_question":
        _questions(backend, thread_id, values, pause)
    elif pause and pause["type"] == "formative_check":
        _formative(backend, thread_id, pause)
    elif pause and pause["type"] == "post_test_and_feedback":
        _posttest(backend, thread_id, values, pause)
    elif snap["finished"]:
        _summary(backend, thread_id, values)
    else:
        st.error(f"Unexpected agent state: next={snap['next']}, interrupt={pause}")
