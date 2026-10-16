# Observability with Langfuse

The app is traced with the **Langfuse Python SDK v5** (`langfuse==5.0.1`) through a single module,
`telemetry/tracing.py`. The instrumentation follows the installed Langfuse skill
(`.claude/skills/langfuse`, `references/instrumentation.md`) and the
[trace best practices](https://langfuse.com/docs/observability/best-practices).

Tracing is **off** until `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, and a tracing failure
never affects a learner's session.

## Setup
```dotenv
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000   # self-hosted; or https://cloud.langfuse.com
LANGFUSE_ENVIRONMENT=development
```

- **Keys:** create them in your Langfuse project under **Settings → API Keys**. EU cloud is `https://cloud.langfuse.com`, US cloud is `https://us.cloud.langfuse.com`, or use your self-hosted URL.
- **Optional settings:** `LANGFUSE_RELEASE`, `LANGFUSE_SAMPLE_RATE` (0–1), and `LANGFUSE_CAPTURE_CONTENT=false` to stop sending learner text and model answers.
- **Restart** Streamlit and uvicorn after editing `.env`. Scripts flush before exiting, and the FastAPI app flushes on shutdown.

## What a trace is

| Trace name | Unit of work | Root input → output |
|---|---|---|
| `start-learning-session` | session start → topic choice or pre-test shown | topic → next prompt |
| `choose-topic` | learner picks a topic → pre-test shown | topic id → next prompt |
| `submit-pretest` | pre-test scored → videos recommended (`recommend-videos` nested) | answers → video choices |
| `choose-video` / `refresh-videos` / `end-session` | video picked and its transcript indexed → context, policy selected | item id → next prompt |
| `answer-question` | one question through CRAG and the LLM | **the question → the answer** |
| `watch-another-video` | learner asks for another video after their doubts | `{"more_videos": true}` → video list (already-watched videos marked) |
| `end-questions` | learner finishes asking and takes the post-test | `{"end": true}` → post-test prompt |
| `submit-posttest` | post-test, evaluation, experience stored | answers, rating, comment → QoE/SRE/reward |
| `recommend-videos` | one YouTube recommendation request from the API or the recommendations page (inside a session it is a child of `submit-pretest`) | topic/objective → ranked videos |

**Correlation fields:**
- Every trace of a learning session shares `session_id` (the learning-session id) and `user_id` (the learner id), so the Sessions view shows the whole session in order.
- Tags mark the feature: `learning-session` or `video-recommendations`.
- `topic_id` is trace metadata.
- The runner also returns each step's `trace_id`.

## Observations inside a trace

| Observation | Type | Where |
|---|---|---|
| `load-learner-profile`, `choose-topic`, `plan-pretest`, `run-pretest`, `receive-video-choice`, `index-video-transcript`, `receive-question`, `rewrite-query`, `route-llm`, `score-posttest`, `store-experience` | span | graph nodes (`agent/node_tracing.py`) |
| `measure-context` | tool | context node (RTT probe, CPU and memory) |
| `select-policy` | chain | policy node; child `retrieve-similar-sessions` (retriever, FAISS) |
| `retrieve-lesson-context` | retriever | CRAG retrieval over the chosen video's index; input names `resource_id` (`yt_<video_id>`) and the video title, output the chunks with scores; child `embed-query` |
| `generate-answer`, `generate-mcqs`, `explain-answers` | **generation** | `services/llm_clients.py`: model, parameters, OpenAI-format messages, token usage (`input`/`output`) for cost |
| `extract-answer` | tool | model-free fallback |
| `embed-query`, `embed-documents` | embedding | `services/vector_store.py` (model, input tokens) |
| `evaluate-session` | evaluator | QoE, SRE, reward, predicted vs observed utility |
| `select-strategy`, `search-youtube`, `fetch-video-metadata`, `fetch-transcript`, `rank-videos` | chain / tool / span | recommendation pipeline (`fetch-transcript` also under `index-video-transcript`) |

Inputs and outputs are short, readable summaries, never the whole agent state.

**Graph interrupts** (the agent waiting for the learner) are recorded as `waiting_for_learner`, not as errors. The paused node runs again, and is traced again, in the next step's trace.

**Failed calls** (an LLM error, a YouTube error) mark their observation `ERROR` with the message. Recovered failures, such as a missing transcript, are `WARNING`.

**Threads:** transcript fetches run in a thread pool, and each task is bound to a copy of the caller's context (`tracing.with_current_context`). Without that, they would appear as orphan traces with no session or user.

## Scores (named after the observed signal)

| Score | Level | Type | Source |
|---|---|---|---|
| `session_rating` | session | numeric 1–5 | learner rating with the post-test |
| `posttest_score` | session | numeric 0–1 | post-test |
| `learning_gain` | session | numeric 0–1 | normalised Hake gain (only when defined) |
| `session_qoe`, `session_sre`, `adaptation_reward` | session | numeric | `metrics/` evaluation |
| `video_opened`, `video_marked_helpful`, `video_marked_not_relevant`, `video_completed_self_reported` | trace (`recommend-videos`) | boolean | learner buttons. Opening is not watching, and completion is self-reported |
| `video_rating` | trace | numeric 1–5 | star rating of a video |

## Privacy
- **Always on:** export-stage masking (`mask_otel_spans`) redacts emails, Indian mobile numbers and 12-digit Aadhaar-like numbers in every exported input, output, metadata and status message.
- **`LANGFUSE_CAPTURE_CONTENT=false`** replaces every observation's input and output with `"[content capture disabled]"`. Structure, models, token usage, latencies, levels and scores are still sent.
- **Never traced:** API keys (YouTube, Hugging Face) are not part of any payload. Learner names are not traced; only ids are.

## LQ from Langfuse (LLM response quality)
QoE's LQ comes from an LLM-as-a-judge evaluator that Langfuse runs on every `generate-answer` generation. At session
end `telemetry/answer_quality.py` reads those scores back, maps them to 0–1 and averages them per answer. It is
unavailable (weights renormalised) when tracing is off, Langfuse is unreachable, or nothing is scored yet.

1. **Judge model:** Langfuse → Settings → LLM Connections → add an OpenAI-compatible connection. For the local
   Ollama use base URL `http://host.docker.internal:11434/v1` (Langfuse runs in Docker, so not `127.0.0.1`) and any
   API key, for example `ollama`.
2. **Evaluator:** Evaluators → set up an LLM-as-a-judge evaluator (for example relevance or helpfulness) that
   targets observations named `generate-answer`. Map the question to the observation input and the answer to its
   output, and name the score `answer_quality`.
3. **App settings** (`.env`, all optional):
   ```dotenv
   LQ_SCORE_NAMES=["answer_quality"]   # several names are averaged per answer
   LQ_SCORE_MIN=0                      # the judge's scale, e.g. 1 and 5 for a 1-5 judge
   LQ_SCORE_MAX=1
   LQ_WAIT_S=0                         # seconds to wait at session end for evaluators still running
   LQ_SOURCE=langfuse                  # off: LQ stays unavailable
   ```

Evaluators run asynchronously, so the last answers of a session may not be scored when the session ends. The
summary page and the `evaluate-session` span show how many answers LQ covers (`lq.scored` of `lq.answers`).

## Verifying (run once credentials are set)
```powershell
.venv\Scripts\python scripts\run_session.py --mock-llm      # or a real session / the Streamlit app
npx langfuse-cli api observations --help                    # then list observations of the printed trace ids
```
Open the trace in Langfuse and check the tree against the tables above. `tests/test_tracing.py` checks the same structure offline with an in-memory exporter; every other test runs with tracing off (`tests/conftest.py`), so `pytest` never writes to your Langfuse project.

## Next steps (not done)
- **Prompt management:** the answer and MCQ prompts still live in code. Moving them to Langfuse prompt management and linking them to generations would show which prompt version produced each answer (skill reference: `prompt-migration.md`).
- **Evaluators:** once real traces exist, set up evaluators, for example answer groundedness on `generate-answer` (skill reference: `setting-up-evals.md`).
