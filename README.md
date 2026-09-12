# QoE-Aware Self-Adaptive Hybrid Edge–Cloud Educational RAG Assistant

This project is a modular FastAPI application for a personalized educational assistant that combines:

- learner personalization
- educational resource recommendation
- knowledge preparation and retrieval (RAG)
- hybrid local/cloud language-model routing
- speech input/output support
- monitoring and QoE evaluation
- experience memory for adaptive learning

## Current project state

The repository is no longer limited to Phase 1. The codebase currently includes the implemented work described in `IMPLEMENTATION_STATUS.md`, including:

- Phases 1–4: foundation, text Q&A, personalization, and resource recommendation
- Phase 3 expanded personalization: learner preference collection for language, competency, interests, pace, explanation style, detail level, and content preference
- Phase 5: knowledge preparation, chunking, embeddings, FAISS indexing, and retrieval
- Phase 6: RAG-based educational question answering
- Phase 7: CRAG-quality gating and hybrid routing scaffolding
- Phase 8: speech input/output orchestration
- Phase 9: monitoring and QoE evaluation
- Phase 10: persistent experience memory records

The remaining planned roadmap is:

- Phase 11: adaptive intelligence and predefined action selection
- Phase 12: hybrid edge/cloud routing and fallback decisions
- Phase 13: local model hosting and side-by-side evaluation against cloud inference

This project also explicitly covers the two requested changes from ma'am:

1. More learner preference questions are stored in the learner profile and included in `PersonalizationContext`.
2. Token/API exhaustion is handled through safe learner-facing errors, monitoring of provider failures, and a future local fallback path.

## Key modules

- `app/main.py` — FastAPI app factory and route registration
- `app/api/routes/` — HTTP endpoints for interactions, personalization, resources, knowledge, monitoring, and speech
- `app/services/` — business logic for assistant orchestration, personalization, recommendations, knowledge retrieval, QoE, and experience memory
- `app/db/` — SQLAlchemy models, session setup, repositories, and database initialization
- `app/knowledge/` — transcript providers, embedding providers, and FAISS index store
- `app/llm/` — LLM provider abstractions and concrete providers
- `app/youtube/` — YouTube Data API integration for resource discovery
- `tests/` — unit and integration tests

## Local setup

1. Create and activate a virtual environment if needed.
2. Install project dependencies:

```bash
python -m pip install -e ".[test]"
```

3. Copy `.env.example` to `.env` and configure your local settings.

4. Run the application:

```bash
uvicorn app.main:create_app --factory --reload
```

## Running tests

```bash
pytest
```

## Configuration

Configuration is loaded from environment variables and optional `.env` values. See `.env.example` for the available settings, including:

- data and database paths
- Hugging Face model/provider settings
- YouTube API settings
- knowledge chunking and embedding parameters
- retrieval and CRAG thresholds
- speech provider toggles

## Important notes

- The application uses a local SQLite-first setup by default.
- Knowledge/RAG features depend on the availability of vector and embedding tooling in the selected Python environment.
- The speech routes use passthrough providers unless you wire in real STT/TTS implementations.
- The project is intentionally modular and layered; see `architecture.md` and `IMPLEMENTATION_STATUS.md` for the design and phase details.

## Repository docs

- `architecture.md` — system design and phased roadmap
- `IMPLEMENTATION_STATUS.md` — implemented vs. deferred work
- `FILE_METADATA.md` — file-by-file metadata catalog
