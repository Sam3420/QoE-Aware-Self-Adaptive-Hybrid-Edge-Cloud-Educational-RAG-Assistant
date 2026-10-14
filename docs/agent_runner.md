# Agent runner API (for the UI / API layer)

`agent/runner.py` is the only entry point the UI/API needs. It has no FastAPI or Streamlit code.

```python
from agent.runner import start_session, resume, get_state

start_session(student_id: str, topic_id: str = "") -> {"thread_id": str, "interrupt": dict}
resume(thread_id: str, payload: dict | str)   -> {"thread_id": str, "interrupt": dict}
                                               | {"thread_id": str, "final_state": dict}
get_state(thread_id: str) -> {"thread_id", "values", "next": [node, ...], "interrupt": dict | None, "finished": bool}
```

- `thread_id` is the `LearningSession.id` that `start_session` creates in the app DB. The same id is
  the LangGraph checkpoint thread and the `session_id` of the stored experience.
- The learner must already exist (`learner_profiles`). Otherwise `start_session` raises
  `agent.interfaces.LearnerNotFound`.
- `resume` on a finished session raises `agent.runner.SessionNotPaused`. An unknown `thread_id`
  raises `agent.runner.SessionNotFound`.
- Checkpoints are stored in `data/checkpoints.db`, separate from `data/app.db`. A session survives a
  server restart and can be resumed with the same `thread_id`.

## Interrupts

Every interrupt payload has a `"type"`. If a resume payload fails validation, the **same** interrupt
comes back with an extra `"error"` string (the session does not crash). Show the error and ask again.

| type | Payload sent to the UI | Resume payload | What happens next |
|---|---|---|---|
| `select_topic` | `{"type", "topics": [{topic_id, title, label, grade, subject}], "prompt"}` (skipped when `start_session` is given a topic) | `{"topic_id": "<catalogue id>"}` | The topic's lesson-note index is built if missing (it grounds the topic-level pre/post-test), then `plan_pretest` runs. An id outside `knowledge_prep/catalog.py` is re-asked |
| `pre_test` | `{"type", "items": [{question_no, item_id, question, options[4], concept, reason}], "plan": {mode, weak_concepts, previous, notes}, "prompt"}` | `{"answers": ["A", "C", ...]}`, one per item, letters A–D or indices 0–3 | `submit_pretest` scores the shown items, then `pre_score` and `competency` are set. Every session has a pre-test: `plan_pretest` gives a new learner the standard first questions, and a returning learner the questions they missed last time plus unseen ones on their weak concepts (generating new questions into the bank when needed). The post-test asks the same items |
| `select_video` | `{"type", "topic_id", "items": [{item_id, rank, video_id, title, url, channel_title, duration_s, score, explanation, evidence_level, caption_language, caption_type, thumbnail_url, unavailable}], "status", "errors", "strategy_id", "decision_reason", "notice", "prompt"}` | `{"item_id": "..."}` to learn from a video; `{"refresh": true}` for new recommendations (the shown ones are recorded as dismissed); `{"end": true}` to go to the post-test | `recommend_videos` searched YouTube with the topic, the competency from the pre-test and the learner's language (strategy R1–R3 by kNN). A chosen video already indexed is reused; otherwise `prepare_knowledge` fetches its transcript in the learner's language, chunks and embeds it into `yt_<video_id>`, and stores the video as an educational resource. If the transcript is unavailable the next recommendation is tried; if none works, `select_video` comes back with `notice` and the failed items marked `unavailable`. Questions are then answered from the video's index |
| `ask_question` | `{"type", "question_no", "prompt"}` | `{"query": "..."}` (or a bare string) to ask; `{"end": true}` to finish | A query goes through CRAG, the router and generation, then back to `ask_question`. `end` goes to `post_test_and_feedback`. The cap is `MAX_QUESTIONS_PER_SESSION` |
| `post_test_and_feedback` | `{"type", "items": [...same shape...], "prompt"}` | `{"answers": [...], "rating": 1-5 or null, "comment": str or null}` | `submit_posttest_and_feedback` runs, then `evaluate_session` and `store_experience`, and the session ends |

The latest answer is in `get_state(thread_id)["values"]["response"]`. Read it after each
`ask_question` interrupt returns.

## `final_state` (main keys)

| key | meaning |
|---|---|
| `policy_id`, `runtime_config` | chosen policy and `{top_k, threshold, max_retries, route}` |
| `decision_reason` | for example `"P2: mean reward 0.85 over 2 of 3 neighbours"`, `"cold start: ... → default P1"`, `"exploration (ε=0.2): ..."` |
| `similar_exps` | kNN neighbours: `{session_id, policy_id, reward, distance}` |
| `context` | `{rtt_ms, cpu, mem, competency_num, last_qoe}` |
| `pre_score`, `post_score`, `competency`, `rating`, `comment` | assessment and feedback |
| `qa_log` | one entry per question: `{question_no, query, retrieval_query, retrieval_score, crag_retries, n_chunks, preferred_route, model_used, latency_ms, gen_latency_ms, error}` |
| `evaluation`, `qoe`, `sre`, `reward` | session evaluation (**stub** until the metrics exist) |
| `session_errors` | non-fatal problems: probe failure, pre-test generation failure, storage failure |

## Minimal driver

```python
out = start_session(student_id, "photosynthesis")
while "interrupt" in out:
    payload = ask_user(out["interrupt"])          # your UI
    out = resume(out["thread_id"], payload)
show(out["final_state"])
```
