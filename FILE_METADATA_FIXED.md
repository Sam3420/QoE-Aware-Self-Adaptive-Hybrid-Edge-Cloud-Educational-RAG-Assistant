# Project File Metadata

This document provides a concise metadata catalog for the current repository snapshot. It is intended to help with navigation, ownership awareness, implementation traceability, and module-level understanding across the project.

## Metadata Schema

Each entry contains:
- Path
- Type
- Phase
- Purpose
- Inputs
- Outputs
- Key dependencies
- Status

---

## Root-Level Files

- `.env.example` — Type: configuration template; Phase: 1–10; Purpose: example environment variable file for local development and deployment configuration; Inputs: developer environment settings and deployment preferences; Outputs: `.env` values consumed by runtime settings; Key dependencies: `app/core/config.py`, local environment; Status: present.
- `.gitignore` — Type: repository housekeeping; Phase: n/a; Purpose: excludes local environment, caches, generated artifacts, and temporary data from version control; Inputs: repository state; Outputs: clean Git working tree; Key dependencies: workspace environment; Status: present.
- `.vscode/settings.json` — Type: editor configuration; Phase: n/a; Purpose: workspace/editor settings for the project; Inputs: editor preferences; Outputs: interpreter/test discovery and editor behavior; Key dependencies: VS Code configuration; Status: present.
- `README.md` — Type: project documentation; Phase: 1–10; Purpose: user-facing overview, setup instructions, and repository orientation; Inputs: project state and architecture summary; Outputs: contributor onboarding guidance; Key dependencies: `architecture.md`, `IMPLEMENTATION_STATUS.md`; Status: present.
- `architecture.md` — Type: architecture documentation; Phase: 1–10; Purpose: system design, module breakdown, data model, and phase roadmap; Inputs: implementation decisions and design constraints; Outputs: architectural understanding for developers; Key dependencies: implemented code modules; Status: present.
- `IMPLEMENTATION_STATUS.md` — Type: status document; Phase: 1–10; Purpose: tracks implemented, deferred, and completed phases; Inputs: code changes and feature completion; Outputs: a summary of current phase status; Key dependencies: architecture roadmap and codebase state; Status: present.
- `pyproject.toml` — Type: Python packaging/configuration; Phase: 1–10; Purpose: project metadata, dependencies, and pytest configuration; Inputs: package metadata and runtime requirements; Outputs: installable package configuration and test settings; Key dependencies: Python package ecosystem; Status: present.
- `pytest_unit_out.txt` — Type: verification artifact; Phase: verification; Purpose: saved stdout from earlier unit-test execution; Inputs: test run output; Outputs: verification log for review/debugging; Key dependencies: pytest environment; Status: generated artifact.
- `test-output.txt` — Type: verification artifact; Phase: verification; Purpose: saved stdout from earlier project/test runs; Inputs: command output; Outputs: verification log for troubleshooting; Key dependencies: runtime/test environment; Status: generated artifact.
- `data/` — Type: runtime data directory; Phase: 1–10; Purpose: local storage area for generated artifacts, indexes, metrics, and other runtime outputs; Inputs: application usage/data generation; Outputs: persisted model, vector, and experiment artifacts; Key dependencies: settings-driven storage paths; Status: present.
- `qoe_educational_assistant.egg-info/` — Type: generated packaging metadata; Phase: n/a; Purpose: package metadata produced by editable install/build processes; Inputs: build/install state; Outputs: distribution metadata files; Key dependencies: Python packaging tools; Status: present.

---

## Application Package

### `app/__init__.py`
- Type: package marker
- Phase: 1
- Purpose: marks the `app` package as importable
- Inputs: Python package discovery
- Outputs: importable package namespace
- Key dependencies: Python package structure
- Status: present

### `app/main.py`
- Type: application entrypoint
- Phase: 1–10
- Purpose: creates the FastAPI application, registers APIs, and sets up app startup behavior
- Inputs: settings, route modules, DB initialization hooks
- Outputs: `FastAPI` app instance and OpenAPI docs
- Key dependencies: `app.api.routes.*`, `app.db.init_db`, `app.core.config`
- Status: implemented

### `app/api/`

- `app/api/__init__.py` — Type: package marker; Phase: 1; Purpose: package initializer; Inputs: Python package discovery; Outputs: API package namespace; Key dependencies: none; Status: present.
- `app/api/dependencies.py` — Type: dependency wiring; Phase: 1–10; Purpose: provides DB sessions and service/provider instances for FastAPI routes; Inputs: request context and settings; Outputs: service/provider objects consumed by routes; Key dependencies: DB session, services, providers; Status: implemented.
- `app/api/routes/health.py` — Type: health endpoint; Phase: 1–2; Purpose: exposes readiness/liveness checks; Inputs: HTTP request; Outputs: status JSON; Key dependencies: FastAPI; Status: implemented.
- `app/api/routes/interactions.py` — Type: interaction API; Phase: 2, 5, 6; Purpose: submits text questions and orchestrates assistant responses; Inputs: session ID, question payload, DB session, assistant service; Outputs: question answer and persisted interaction metadata; Key dependencies: `AssistantService`, DB layer; Status: implemented.
- `app/api/routes/knowledge.py` — Type: knowledge API; Phase: 5; Purpose: prepares and retrieves knowledge resources for RAG workflows; Inputs: resource ID, retrieval request, DB session; Outputs: prepared knowledge or ranked retrieval results; Key dependencies: `KnowledgePreparationService`, `KnowledgeRetrievalService`; Status: implemented.
- `app/api/routes/monitoring.py` — Type: monitoring API; Phase: 9; Purpose: returns QoE/monitoring information for interactions; Inputs: interaction ID and DB session; Outputs: QoE score payload; Key dependencies: `MonitoringService`; Status: implemented.
- `app/api/routes/personalization.py` — Type: personalization API; Phase: 3; Purpose: manages learner profiles, preferences, and assessments; Inputs: learner ID, update payloads; Outputs: personalization context and assessment responses; Key dependencies: personalization service and repositories; Status: implemented.
- `app/api/routes/resources.py` — Type: resource API; Phase: 4; Purpose: returns recommendations and resource selection actions; Inputs: session ID, recommendation request; Outputs: ranked resource candidates and selection results; Key dependencies: `ResourceRecommendationService`, YouTube provider; Status: implemented.
- `app/api/routes/speech.py` — Type: speech API; Phase: 8; Purpose: handles speech-based interaction requests; Inputs: audio upload, runtime config, resource ID; Outputs: transcript, answer text, and audio response; Key dependencies: `SpeechService`, STT/TTS providers; Status: implemented.

### `app/core/`

- `app/core/__init__.py` — Type: package marker; Phase: 1; Purpose: package initializer; Inputs: Python package discovery; Outputs: core package namespace; Key dependencies: none; Status: present.
- `app/core/config.py` — Type: runtime configuration; Phase: 1–10; Purpose: central settings and environment-driven configuration loading; Inputs: `.env` and environment variables; Outputs: `Settings` object and resolved storage paths; Key dependencies: `pydantic-settings`; Status: implemented.
- `app/core/ids.py` — Type: identifier utilities; Phase: 1–10; Purpose: generates UUID-based IDs for domain entities; Inputs: none; Outputs: unique string identifiers; Key dependencies: Python UUID utilities; Status: implemented.
- `app/core/logging.py` — Type: logging setup; Phase: 1–10; Purpose: configures application logging; Inputs: log level and logger names; Outputs: configured logger objects; Key dependencies: Python logging; Status: implemented.

### `app/domain/`

- `app/domain/__init__.py` — Type: package marker; Phase: 1; Purpose: package initializer for domain layer; Inputs: Python package discovery; Outputs: domain package namespace; Key dependencies: none; Status: present.
- `app/domain/enums.py` — Type: enums; Phase: 1–10; Purpose: common domain enums for session, interaction, language, and competency states; Inputs: application concepts; Outputs: typed enum values; Key dependencies: services, models, routes; Status: implemented.
- `app/domain/models/__init__.py` — Type: package marker; Phase: 1; Purpose: package initializer for domain models; Inputs: package discovery; Outputs: model namespace; Key dependencies: none; Status: present.
- `app/domain/models/core.py` — Type: Pydantic domain models; Phase: 1–10; Purpose: request/response DTOs and shared typed contracts; Inputs: API payloads and service inputs; Outputs: validated domain models; Key dependencies: routes and services; Status: implemented.

### `app/llm/`

- `app/llm/__init__.py` — Type: package marker; Phase: 1; Purpose: package initializer for LLM layer; Inputs: package discovery; Outputs: LLM namespace; Key dependencies: none; Status: present.
- `app/llm/provider.py` — Type: provider interface; Phase: 2–10; Purpose: exposes the generic LLM provider contract; Inputs: generation request objects; Outputs: provider-generated text results; Key dependencies: concrete providers; Status: implemented.
- `app/llm/types.py` — Type: request/response types; Phase: 2–10; Purpose: shared request/result/error models for providers; Inputs: provider request data; Outputs: structured provider contracts; Key dependencies:(provider implementations) ; Status: implemented.
- `app/llm/huggingface_provider.py` — Type: concrete cloud provider; Phase: 2–10; Purpose: wraps Hugging Face inference for text generation; Inputs: prompt, system prompt, model parameters; Outputs: generated text results; Key dependencies: Hugging Face client, settings; Status: implemented.
- `app/llm/hybrid_provider.py` — Type: hybrid provider abstraction; Phase: 7; Purpose: routes local vs cloud inference decisions; Inputs: request and settings; Outputs: local/cloud-generated result with fallback logic; Key dependencies: local/cloud providers; Status: scaffolded/partial.
- `app/llm/stt_provider.py` — Type: STT adapter interfaces; Phase: 8; Purpose: defines transcription provider contracts; Inputs: audio bytes and MIME type; Outputs: transcript text; Key dependencies: speech routes; Status: implemented.
- `app/llm/tts_provider.py` — Type: TTS adapter interfaces; Phase: 8; Purpose: defines text-to-speech provider contracts; Inputs: plain text; Outputs: synthesized audio bytes; Key dependencies: speech routes; Status: implemented.

### `app/youtube/`

- `app/youtube/__init__.py` — Type: package marker; Phase: 1; Purpose: package initializer; Inputs: package discovery; Outputs: YouTube package namespace; Key dependencies: none; Status: present.
- `app/youtube/provider.py` — Type: provider interface; Phase: 4; Purpose: defines the resource discovery contract; Inputs: search query and result budget; Outputs: list of candidate video metadata dictionaries; Key dependencies: downstream recommendation services; Status: implemented.
- `app/youtube/data_api_provider.py` — Type: concrete provider; Phase: 4; Purpose: queries YouTube Data API v3 for educational video candidates; Inputs: query, language, max results; Outputs: normalized YouTube video metadata; Key dependencies: `httpx`, settings; Status: implemented.

### `app/knowledge/`

- `app/knowledge/__init__.py` — Type: package marker; Phase: 1; Purpose: package initializer; Inputs: package discovery; Outputs: knowledge package namespace; Key dependencies: none; Status: present.
- `app/knowledge/transcript_provider.py` — Type: transcript interface; Phase: 5; Purpose: defines transcript retrieval contract for resources; Inputs: resource ID and resource object; Outputs: transcript text; Key dependencies: knowledge preparation workflow; Status: implemented.
- `app/knowledge/youtube_transcript_provider.py` — Type: concrete transcript provider; Phase: 5; Purpose: fetches YouTube captions/transcripts; Inputs: resource ID and resource metadata; Outputs: normalized transcript text; Key dependencies: `youtube-transcript-api`; Status: implemented.
- `app/knowledge/embedding_provider.py` — Type: embedding provider; Phase: 5; Purpose: wraps sentence embeddings for chunking and retrieval; Inputs: text or batches of text; Outputs: vector embeddings; Key dependencies: `sentence-transformers`; Status: implemented.
- `app/knowledge/faiss_index_store.py` — Type: vector index storage; Phase: 5; Purpose: builds and searches FAISS indexes for knowledge chunks; Inputs: embeddings, chunk IDs, query vectors; Outputs: persisted index metadata and matching chunk IDs; Key dependencies: FAISS, NumPy; Status: implemented.

### `app/services/`

- `app/services/__init__.py` — Type: package marker; Phase: 1; Purpose: package initializer; Inputs: Python package discovery; Outputs: services namespace; Key dependencies: none; Status: present.
- `app/services/assistant_service.py` — Type: orchestration service; Phase: 2, 5, 6, 7; Purpose: handles text Q&A, personalization, retrieval, prompt construction, and interaction persistence; Inputs: session ID, question, runtime configuration, optional resource ID; Outputs: answer payload and stored interaction metadata; Key dependencies: repositories, LLM provider, retrieval services; Status: implemented.
- `app/services/errors.py` — Type: error definitions; Phase: 1–10; Purpose: shared exceptions for service-level failures; Inputs: failure conditions; Outputs: typed errors for callers/tests; Key dependencies: service and route layers; Status: implemented.
- `app/services/experience_service.py` — Type: experience-memory service; Phase: 10; Purpose: persists state/action/config/QoE/reward experiences for adaptive learning; Inputs: interaction identifiers; Outputs: experience records and lookups; Key dependencies: experience models and repositories; Status: implemented.
- `app/services/knowledge_preparation_service.py` — Type: knowledge pipeline; Phase: 5; Purpose: processes transcripts into chunks, embeddings, and indexes; Inputs: resource ID and transcript content; Outputs: stored chunk/index metadata and prepared knowledge; Key dependencies: transcript provider, embedding provider, FAISS store; Status: implemented.
- `app/services/knowledge_retrieval_service.py` — Type: retrieval service; Phase: 5, 6; Purpose: searches prepared knowledge indexes for relevant context; Inputs: resource ID, query, top K; Outputs: ranked context chunks; Key dependencies: embedding provider, FAISS store, DB models; Status: implemented.
- `app/services/learning_trace_service.py` — Type: trace service; Phase: 1–10; Purpose: records learning trace artifacts for persistence and analysis; Inputs: raw domain objects or data payloads; Outputs: persisted trace records; Key dependencies: learning trace repository; Status: implemented.
- `app/services/monitoring_service.py` — Type: QoE monitoring service; Phase: 9; Purpose: aggregates runtime metrics and computes QoE scores; Inputs: interaction and observation data; Outputs: metrics and QoE labels/scores; Key dependencies: repositories and DB models; Status: implemented.
- `app/services/personalization_service.py` — Type: personalization service; Phase: 3; Purpose: stores learner profile preferences and assessment results; Inputs: learner data, assessment results, preferences; Outputs: personalization context and persisted profile data; Key dependencies: learner/session models; Status: implemented.
- `app/services/resource_recommendation_service.py` — Type: recommendation service; Phase: 4; Purpose: ranks educational resources for learners; Inputs: session context, recommendation request; Outputs: ranked resource candidates; Key dependencies: YouTube provider and personalization state; Status: implemented.
- `app/services/speech_service.py` — Type: speech orchestration service; Phase: 8; Purpose: coordinates STT/TTS and assistant answer generation; Inputs: audio bytes, MIME type, session info; Outputs: transcript, answer text, and synthesized audio; Key dependencies: providers and assistant service; Status: implemented.

### `app/db/`

- `app/db/__init__.py` — Type: package marker; Phase: 1; Purpose: package initializer; Inputs: package discovery; Outputs: DB package namespace; Key dependencies: none; Status: present.
- `app/db/base.py` — Type: SQLAlchemy base; Phase: 1; Purpose: central ORM declarative base; Inputs: model declarations; Outputs: SQLAlchemy metadata and base classes; Key dependencies: SQLAlchemy; Status: implemented.
- `app/db/errors.py` — Type: database error handling; Phase: 1–10; Purpose: custom exceptions for persistence issues; Inputs: persistence failure conditions; Outputs: typed DB exceptions; Key dependencies: service/repository layers; Status: implemented.
- `app/db/init_db.py` — Type: DB initialization; Phase: 1; Purpose: creates tables and initializes the SQLite schema; Inputs: SQLAlchemy engine and model metadata; Outputs: initialized database tables; Key dependencies: models and engine; Status: implemented.
- `app/db/session.py` — Type: session utilities; Phase: 1; Purpose: supplies SQLAlchemy engine and session factory; Inputs: settings and DB URL; Outputs: engine and session factory; Key dependencies: settings and SQLAlchemy; Status: implemented.

### `app/db/models/`

- `app/db/models/__init__.py` — Type: package marker; Phase: 1; Purpose: package initializer; Inputs: package discovery; Outputs: models namespace; Key dependencies: none; Status: present.
- `app/db/models/assessment_result.py` — Type: ORM model; Phase: 3; Purpose: stores learner assessment outcomes; Inputs: assessment payloads; Outputs: persisted assessment rows; Key dependencies: learner/session relationships; Status: implemented.
- `app/db/models/educational_resource.py` — Type: ORM model; Phase: 4; Purpose: stores resource metadata and source details; Inputs: YouTube metadata; Outputs: resource records; Key dependencies: knowledge documents and sessions; Status: implemented.
- `app/db/models/experience.py` — Type: ORM model; Phase: 10; Purpose: stores experience-memory snapshots and reward outcomes; Inputs: interaction-derived state/action/config snapshots; Outputs: persisted experience records; Key dependencies: learner/session/runtime configuration/resource models; Status: implemented.
- `app/db/models/interaction.py` — Type: ORM model; Phase: 2; Purpose: stores learner interactions and responses; Inputs: questions, answers, metadata; Outputs: persisted interaction rows; Key dependencies: learner/session/resource/configuration; Status: implemented.
- `app/db/models/learner.py` — Type: ORM model; Phase: 1, 3; Purpose: stores learner profiles and personalization attributes; Inputs: learner profile data; Outputs: learner rows; Key dependencies: sessions, assessments, interactions; Status: implemented.
- `app/db/models/learning_session.py` — Type: ORM model; Phase: 1; Purpose: stores learning-session state and related context; Inputs: learner/session state; Outputs: session rows; Key dependencies: learner, interactions, runtime configuration; Status: implemented.
- `app/db/models/runtime_configuration.py` — Type: ORM model; Phase: 1–10; Purpose: stores immutable runtime configuration snapshots; Inputs: configuration data snapshots; Outputs: persisted runtime configuration history; Key dependencies: interactions and experiences; Status: implemented.

### `app/db/repositories/`

- `app/db/repositories/__init__.py` — Type: package marker; Phase: 1; Purpose: package initializer; Inputs: package discovery; Outputs: repository namespace; Key dependencies: none; Status: present.
- `app/db/repositories/learning_trace.py` — Type: repository module; Phase: 1–10; Purpose: persistence layer for learning trace and related records; Inputs: domain objects or raw row data; Outputs: CRUD results across core tables; Key dependencies: DB session and models; Status: implemented.

### `tests/`

- `tests/__init__.py` — Type: package marker; Phase: n/a; Purpose: test package initializer; Inputs: Python package discovery; Outputs: test package namespace; Key dependencies: pytest discovery; Status: present.
- `tests/unit/test_assistant_service.py` — Type: unit test; Phase: 2, 5, 6; Purpose: checks assistant orchestration and answer generation behavior; Inputs: fake providers and DB/session data; Outputs: interaction and prompt assertions; Key dependencies: assistant service; Status: present.
- `tests/unit/test_config.py` — Type: unit test; Phase: 1; Purpose: verifies configuration loading and defaults; Inputs: environment values and monkeypatching; Outputs: settings assertions; Key dependencies: config module; Status: present.
- `tests/unit/test_huggingface_provider.py` — Type: unit test; Phase: 2; Purpose: validates provider request packaging and result handling; Inputs: fake requests/responses; Outputs: provider behavior assertions; Key dependencies: Hugging Face provider; Status: present.
- `tests/integration/test_learning_trace_persistence.py` — Type: integration test; Phase: 1–10; Purpose: verifies persistence of learning-trace records across DB-backed flows; Inputs: session/resource/configuration records; Outputs: persisted trace assertions; Key dependencies: repository and DB models; Status: present.
- `tests/integration/test_text_question_api.py` — Type: integration test; Phase: 2, 6; Purpose: exercises the text-question API end-to-end; Inputs: HTTP request payload; Outputs: API response and persisted interaction; Key dependencies: FastAPI app, services, DB layer; Status: present.

---

## Notes

- This metadata file is intended as a compact project map, not a replacement for detailed code documentation.
- The detailed function-level catalog is kept in `FILE_METADATA1.md`.
- The file now explicitly includes the main Input and Output contract for each entry.
- Some files are scaffolding, package markers, or generated artifacts rather than functional source modules.
