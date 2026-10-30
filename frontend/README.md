# Streamlit frontend

A presentation layer over the existing backend. Every button runs real backend code: the FastAPI
API, the LangGraph agent, FAISS/CRAG retrieval, psutil/RTT monitoring and the app database. Nothing
is simulated. When a component is unavailable, the UI shows the real error instead of output.

## How it connects to the backend

| What | How | Why |
|---|---|---|
| Backend health, text questions (Phase 2 assistant) | HTTP → FastAPI (`GET /health`, `POST /sessions/{id}/interactions/text-question`) | These are the only routes the API exposes |
| Learning sessions (topic → pre-test → video recommendation → video choice → transcript index → policy → CRAG → router → post-test → experience) | In-process → `agent/runner.py` `SessionRunner` with `agent.deps.make_real_services()` | The agent has no HTTP API; `docs/agent_runner.md` names the runner as the UI entry point |
| Learners, sessions, personalization, history | In-process → `AppGateway`, `LearningTraceRepository`, `AdaptationRepository` | No API routes exist for them |
| Index building, retrieval explorer | In-process → `knowledge_prep.build_index`, `services.crag.CragRetriever` | Same objects the agent uses |
| CPU / memory / RTT | In-process → `services.monitoring.SystemMonitor` | Same measurements `collect_context` uses |

The frontend never installs the fake services from `services/fakes.py`. No backend code was changed.

## Run it

All commands run from the **repository root**. The API and the frontend both resolve `.env` and
`data/app.db` against the working directory, so they must share one.

```powershell
# one-time setup
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt fastapi uvicorn

# terminal 1: FastAPI backend (creates tables on startup)
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# terminal 2: Streamlit frontend
.venv\Scripts\python -m streamlit run frontend\streamlit_app.py
```

Use `python -m …` rather than `.venv\Scripts\uvicorn.exe` / `streamlit.exe`: Windows Smart App Control can block
those pip-generated launchers ("Permission denied"), while python.exe is signed.

Open http://localhost:8501. The database tables and policies P1–P5 are created on first start; no
separate initialisation command is needed.

### Optional: local LLM and embeddings (Ollama)

The agent's local route and its default embedder use [Ollama](https://ollama.com):

```powershell
ollama pull gemma3:1b          # local (edge) LLM route
ollama pull nomic-embed-text   # embeddings for FAISS retrieval
```

## Configuration (`.env` in the repository root)

| Variable | Used by | Notes |
|---|---|---|
| `HF_TOKEN` | Speech-to-text model download | Only to download the gated SraVaani model; no LLM runs on Hugging Face. |
| `EMBEDDING_BACKEND` | Agent retrieval, index building | `ollama` (default, needs Ollama + `nomic-embed-text`) or `hashing` (model-free; see limitation below) |
| `GROQ_API_KEY` | API assistant, agent cloud route, MCQ generation | **Required for any cloud LLM** (Groq). |
| `GROQ_MODEL` / `LLM_MODEL_ID` | Agent cloud route + MCQs / API assistant | default `openai/gpt-oss-120b`; the API assistant uses `GROQ_MODEL` unless `LLM_MODEL_ID` is set |
| `LOCAL_LLM_MODEL`, `OLLAMA_BASE_URL` | Agent local route | default `gemma3:1b`, `http://127.0.0.1:11434` |
| `DATABASE_PATH` / `DATABASE_URL` | Both | default `data/app.db` |
| `YOUTUBE_API_KEY` | Video recommendations | YouTube Data API v3 key; see docs/recommendations.md |
| `QOE_API_URL` | Frontend only (new) | default `http://127.0.0.1:8000` |
| `QOE_API_TIMEOUT_S` | Frontend only (new) | default `75`; HTTP timeout for a text question |

Restart both processes after editing `.env`; settings are read once at startup.

## Pages

- **Dashboard**: live API/DB/LLM-route/index status and activity counts from the DB and checkpoints.
- **Learning session**: the full agent loop. Topic choice, pre-test, video choice, chat and post-test are each a real graph interrupt. Videos come from the YouTube Data API (needs `YOUTUBE_API_KEY`); the chosen video's transcript is fetched in the learner's language, indexed once as `yt_<video_id>`, and answers are retrieved from it. A video without a usable transcript is skipped for the next recommendation. Each answer has a *Technical details* panel (model, route, latencies, retrieval score, CRAG retries, retrieved chunks, errors).
- **Quick ask (API)**: the FastAPI assistant over HTTP. The conversation is read back from the `interactions` table, including failed attempts the backend recorded.
- **Learner profile**: Phase 3 fields saved to `learner_profiles`, with notes on which fields the agent actually reads.
- **Knowledge & retrieval**: index status, build/rebuild, transcript upload, and a CRAG retrieval explorer.
- **Video recommendations**: YouTube search + transcript evidence ranked for the learner, with feedback buttons (watch, helpful, not relevant, finished, stars) and outcome/QoE-signal panels. Needs `YOUTUBE_API_KEY`; see `docs/recommendations.md`.
- **Monitoring**: on-demand psutil CPU/memory and an RTT probe, route health, and recorded latencies.
- **Adaptation & memory**: the policy catalogue, the current kNN decision, and stored experiences.
- **History**: every session of the learner, replayed from checkpoints or the interactions table. Unfinished agent sessions can be continued.
- **System capabilities**: implemented, stub and missing features, with the file that proves each.

The learner, session and page ids are kept in the URL, so a browser refresh restores the same
session from the database and checkpoints.

## Known limitations (backend, not hidden by the UI)

- **QoE/SRE/reward** are computed by `metrics/` (see `docs/adaptive_policy.md`). LLM-response, STT and TTS quality have no data source and are reported as unavailable; sentiment and energy are labelled estimates.
- **Speech-to-text and text-to-speech are not implemented** in the code (despite the main README), so the UI does not offer them.
- **`EMBEDDING_BACKEND=hashing` does not give meaningful retrieval.** Measured on the photosynthesis index, an off-topic question ("capital of France", 0.51) outscores on-topic ones ("chlorophyll", 0.10). With hashing, most questions retrieve no chunks and the model is told the lesson may not cover them. Use Ollama + `nomic-embed-text` for a real RAG demo. With it, on-topic questions scored 0.71–0.82 and off-topic ones 0.47–0.55 on the same index, against the 0.65 threshold.
- Without an LLM, MCQ pre/post-tests cannot be generated. The agent then skips them, and its answers come from the model-free extractive fallback. The UI says so on every such answer.
- Agent Q&A lives in the LangGraph checkpoints (`data/checkpoints.db`), not the `interactions` table. Only the FastAPI route writes `interactions`.
