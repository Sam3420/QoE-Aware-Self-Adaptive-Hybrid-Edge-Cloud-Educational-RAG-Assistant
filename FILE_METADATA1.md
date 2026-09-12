# Project File Metadata (Detailed / Function-Level)

Project: **QoE-Aware Self-Adaptive Hybrid Edge-Cloud Educational RAG Assistant**
Branch snapshot: `feature/knowledge-rag`

This document is a project metadata catalog for the current repository state. It is intended
to be a code map rather than a user-facing README: for each Python module it lists the
classes, methods, and functions defined in the file, plus what each one is responsible for.
Non-code files (docs, config) keep a purpose-level description.

Overall shape of the app: a **FastAPI** service backed by **SQLAlchemy/SQLite**, with a
layered architecture — `api` (HTTP routes) → `services` (business logic/orchestration) →
`db.repositories` (persistence) → `db.models` (ORM tables), plus side packages for `llm`
(cloud/local/hybrid LLM providers), `knowledge` (RAG pipeline: transcripts → chunks →
embeddings → FAISS index), and `youtube` (resource discovery). The app tracks every
interaction's Quality of Experience (QoE) and stores it as reusable "experience" records —
the substrate for the "self-adaptive" phase-10 goal (config/QoE/reward logging for future
policy learning).

---

## Root-Level Files

| File | Type | Purpose |
|---|---|---|
| `.env.example` | config template | Example environment variables: app name/env, log level, data/storage paths, HF token & model id, YouTube API key, chunking/embedding/retrieval knobs, local-LLM & CRAG thresholds, STT/TTS toggles. Copy to `.env` for local runs. |
| `.gitignore` | repo housekeeping | Excludes `.env`, virtualenvs, `__pycache__`, DB files, FAISS/index artifacts, and other generated output from git. |
| `.vscode/settings.json` | editor config | Workspace settings (Python interpreter/test discovery, formatting) for VS Code contributors. |
| `README.md` | docs | Short project overview and setup/run instructions; entry point for new contributors. |
| `architecture.md` | docs | The system-design source of truth: layered architecture, data model, the 10-phase roadmap (foundation → personalization → resources → RAG knowledge → orchestration/CRAG-hybrid routing → speech → monitoring/QoE → experience memory), and rationale for design decisions. |
| `IMPLEMENTATION_STATUS.md` | docs | Short running log of which phases are implemented vs. deferred/partial (e.g., hybrid local/cloud routing is scaffolded). |
| `pyproject.toml` | packaging | Project metadata, runtime + dev dependencies (FastAPI, SQLAlchemy, pydantic-settings, sentence-transformers, faiss, httpx, youtube-transcript-api, pytest, etc.), and pytest configuration. |
| `pytest_unit_out.txt` | generated artifact | Saved stdout from a previous unit-test run (verification log, not source). |
| `test-output.txt` | generated artifact | Saved stdout from a previous full test/run command (verification log, not source). |

---

## `app/` — Application Package

### `app/__init__.py`
Package marker. Module docstring: *"Application package for the Phase 1 foundation."*

### `app/main.py`
FastAPI application factory / entrypoint.
- **`create_app() -> FastAPI`** — builds the `FastAPI` instance, registers all route modules from `app/api/routes/` (health, interactions, knowledge, monitoring, personalization, resources, speech), and wires up OpenAPI/Swagger. This is what an ASGI server (e.g. `uvicorn app.main:create_app`) serves.

---

## `app/api/` — HTTP Layer

### `app/api/__init__.py`
Package marker. Docstring: *"HTTP API package."*

### `app/api/dependencies.py`
FastAPI dependency-injection wiring — constructs and hands routes their service objects (keeps route handlers thin and testable).
- **`get_db_session() -> Generator[Session, None, None]`** — yields a scoped SQLAlchemy `Session` per request, closing it afterward.
- **`get_llm_provider(settings: Settings) -> LLMProvider`** — builds the configured LLM provider (Hugging Face cloud, local, or hybrid) from settings.
- **`get_resource_recommendation_service(db_session, settings) -> ResourceRecommendationService`** — wires the repository + YouTube provider into the recommendation service.
- **`get_knowledge_preparation_service(db_session, settings) -> KnowledgePreparationService`** — wires transcript provider + embedding provider + FAISS store + repository for the ingestion pipeline.
- **`get_knowledge_retrieval_service(db_session, settings) -> KnowledgeRetrievalService`** — wires embedding provider + FAISS store + repository for query-time retrieval.
- **`get_assistant_service(db_session, llm_provider, settings, knowledge_retrieval_service) -> AssistantService`** — assembles the top-level orchestration service (LLM + personalization + retrieval + CRAG quality gate).
- **`get_speech_service(db_session, assistant_service, settings) -> SpeechService`** — wires STT/TTS providers around the assistant service for audio interactions.
- **`get_monitoring_service(db_session) -> MonitoringService`** — wires the repository into the QoE/monitoring service.

### `app/api/routes/__init__.py`
Package marker. Docstring: *"API route modules."*

### `app/api/routes/health.py`
- **`health_check() -> dict[str, str]`** — `GET` liveness endpoint returning a simple status payload; used for uptime/readiness checks.

### `app/api/routes/interactions.py` (Phase 2, 5, 6)
- **`submit_text_question(session_id, request: TextQuestionRequest, db_session, assistant_service) -> TextQuestionResponse`** — main Q&A endpoint. Delegates to `AssistantService.answer_text_question`, persists the interaction, and returns the generated answer with latency/provider/model metadata.

### `app/api/routes/knowledge.py` (Phase 5)
- **`prepare_knowledge(resource_id, db_session, preparation_service) -> KnowledgePreparationResponse`** — triggers the RAG ingestion pipeline (transcript → chunk → embed → index) for a given educational resource.
- **`retrieve_knowledge(resource_id, request: KnowledgeRetrievalRequest, db_session, retrieval_service) -> KnowledgeRetrievalResponse`** — runs a similarity search against a resource's prepared knowledge index and returns ranked chunk hits.

### `app/api/routes/monitoring.py` (Phase 9)
- **`get_interaction_qoe(interaction_id, db_session, monitoring_service) -> QoEScoreResponse`** — returns the computed QoE score/quality label/latency for a past interaction.

### `app/api/routes/personalization.py` (Phase 3)
- **`get_learner_personalization(learner_id, db_session) -> LearnerPersonalizationProfile`** — fetches a learner's stored preferences/competency/interests.
- **`update_learner_personalization(learner_id, request: LearnerPersonalizationUpdate, db_session) -> LearnerPersonalizationProfile`** — updates language/competency/interests/learning-style preferences.
- **`record_assessment_result(learner_id, request: AssessmentResultCreate, db_session) -> AssessmentResultResponse`** — records a topic assessment (competency level + score) for a learner.

### `app/api/routes/resources.py` (Phase 4)
- **`get_resource_recommendations(session_id, request: ResourceRecommendationRequest, db_session, recommendation_service) -> ResourceRecommendationResponse`** — returns ranked YouTube video recommendations for a topic, personalized to the learner.
- **`select_resource(session_id, resource_id, db_session, recommendation_service) -> ResourceSelectionResponse`** — marks a specific recommended resource as selected/active for the session.

### `app/api/routes/speech.py` (Phase 8)
- **`async submit_speech_question(session_id, audio_file: UploadFile, runtime_configuration_id, resource_id, db_session, speech_service) -> SpeechQuestionResponse`** — accepts an uploaded audio question, runs STT → assistant answer → TTS, and returns transcript + text answer + base64 audio.

---

## `app/core/` — Cross-Cutting Runtime Helpers

### `app/core/__init__.py`
Package marker. Docstring: *"Core runtime helpers."*

### `app/core/config.py`
Central `pydantic-settings`-based configuration.
- **`class Settings(BaseSettings)`** — declares every environment-driven setting: `app_name`, `app_env`, `log_level`; storage paths (`data_root`, `database_path`/`database_url`, `knowledge_storage_path`, `vector_index_storage_path`, `metrics_storage_path`, `experience_storage_path`, `faiss_storage_path`); LLM settings (`hf_token`, `llm_model_id`, `hf_inference_url`, timeout/max-tokens/temperature); `default_runtime_configuration_name`; YouTube settings (API key, base URL, default language, max results, timeout); knowledge/RAG settings (`knowledge_chunk_size`, `knowledge_chunk_overlap`, `knowledge_embedding_model`, `knowledge_retrieval_top_k`); local/hybrid LLM settings (`local_llm_enabled`, `local_llm_model_id`); CRAG threshold (`crag_quality_min_score`); STT/TTS provider name + enabled flags.
  - **`resolved_database_url(self) -> str`** — returns `database_url` if set, else derives a SQLite URL from `database_path`.
  - **`resolved_data_root(self) -> Path`** — returns the absolute base data directory, creating it if necessary.
  - **`resolved_storage_paths(self) -> dict[str, Path]`** — returns a dict of every resolved storage subdirectory (knowledge, vectors, metrics, experience, FAISS), ensuring each exists.
  - **`_resolve_path(path: Path) -> Path`** — internal helper resolving a possibly-relative path against `data_root`.
- **`get_settings() -> Settings`** — cached/singleton settings loader used throughout the app (typically via `lru_cache`).

### `app/core/ids.py`
- **`new_id() -> str`** — generates a new unique identifier (UUID-based) used for all primary keys across the domain (learners, sessions, interactions, resources, experiences, etc.).

### `app/core/logging.py`
- **`configure_logging(log_level: str) -> None`** — sets up root/application logging (format, level) at startup.
- **`get_logger(name: str) -> logging.Logger`** — returns a named module logger.

---

## `app/domain/` — Enums & Pydantic Domain Models

### `app/domain/__init__.py`
Package marker. Docstring: *"Typed domain models and enums."*

### `app/domain/enums.py`
Shared string enums used across services, models, and API payloads:
- **`SessionStatus(StrEnum)`** — learning-session lifecycle states (e.g. active/ended).
- **`InteractionStatus(StrEnum)`** — interaction outcome states (e.g. success/failed).
- **`PreferredLanguage(StrEnum)`** — supported learner language preferences.
- **`CompetencyLevel(StrEnum)`** — learner competency levels (e.g. beginner/intermediate/advanced), used both for personalization and assessment results.

### `app/domain/models/__init__.py`
Package marker (no content).

### `app/domain/models/core.py`
All the Pydantic request/response/DTO models shared between routes and services (the API contract layer). Grouped by feature:
- **Learner/personalization:** `LearnerProfileCreate`, `LearningPreferences`, `LearnerPersonalizationUpdate`, `LearnerPersonalizationProfile`, `TopicCompetency`, `PersonalizationContext` (aggregated context injected into prompts).
- **Sessions & resources:** `LearningSessionCreate`, `EducationalResourceCreate`, `RuntimeConfigurationCreate`.
- **Assessments:** `AssessmentResultCreate`, `AssessmentResultResponse`.
- **Resource recommendations:** `ResourceRecommendationRequest`, `ResourceRecommendationItem`, `ResourceRecommendationResponse`, `ResourceSelectionResponse`.
- **Knowledge/RAG pipeline:** `KnowledgePreparationRequest`, `KnowledgePreparationResponse`, `KnowledgeRetrievalRequest`, `KnowledgeRetrievalItem`, `KnowledgeRetrievalResponse`.
- **Interactions:** `InteractionCreate`, `OrmModel` (base config for ORM-mode Pydantic models), `TextQuestionRequest`, `TextQuestionResponse`.
- **Speech:** `SpeechQuestionRequest`, `SpeechQuestionResponse`.
- **Monitoring/QoE/Experience:** `QoEScoreResponse`, `ExperienceRecordResponse` (mirrors the `Experience` ORM model: state/action/config snapshots, QoE outcome, reward score/details, outcome label, performance summary).

---

## `app/llm/` — LLM Provider Abstraction

### `app/llm/__init__.py`
Package marker. Docstring: *"LLM provider interfaces and implementations."*

### `app/llm/provider.py`
- **`LLMProvider(Protocol)`** — structural interface every provider implements: **`generate(request: LLMGenerateRequest) -> LLMGenerateResult`**.

### `app/llm/types.py`
Shared request/response/error types for all providers.
- **`LLMGenerateRequest(BaseModel)`** — `prompt`, `system_prompt`, `model_id`, `max_tokens`, `temperature`.
- **`LLMGenerateResult(BaseModel)`** — `text`, `model_provider`, `model_name`.
- **`LLMProviderError(RuntimeError)`** — raised when a provider cannot produce a safe text response (sanitized upstream failures).

### `app/llm/huggingface_provider.py` (cloud provider)
- **`class HuggingFaceInferenceProvider`**
  - **`__init__(self, token, inference_url, timeout_seconds) -> None`** — configures the HF Inference client.
  - **`generate(self, request: LLMGenerateRequest) -> LLMGenerateResult`** — sends a chat/completion request to the Hugging Face Inference API, parses the response, raises `LLMProviderError` on malformed/HTTP-error responses.

### `app/llm/hybrid_provider.py` (Phase 7, scaffolded)
- **`class LocalLLMProvider`**
  - **`__init__(self, model_id) -> None`** — placeholder/local-model configuration.
  - **`generate(self, request: LLMGenerateRequest) -> LLMGenerateResult`** — currently raises `LLMProviderError`; the repository does not yet implement a real local inference backend, so this remains a boundary for future local hosting rather than an active provider.
- **`class HybridLLMProvider`**
  - **`__init__(self, cloud_provider, local_provider, settings) -> None`** — holds the cloud provider plus an optional local provider and routing settings.
  - **`generate(self, request: LLMGenerateRequest) -> LLMGenerateResult`** — if `local_llm_enabled` is false, or no local provider is configured, it uses the cloud provider. When local routing is enabled, it attempts the local provider first and falls back to the cloud provider on `LLMProviderError`. This is a hybrid routing scaffold rather than a fully deployed local model host.

### `app/llm/stt_provider.py` (Phase 8)
- **`STTProvider(Protocol)`** — **`transcribe(self, audio_bytes: bytes, mime_type) -> str`**.
- **`class PassthroughSTTProvider`** — *"Minimal provider boundary for environments without a real STT backend."*
  - **`__init__(self, provider_name, enabled) -> None`**
  - **`transcribe(self, audio_bytes, mime_type) -> str`** — no-op/placeholder transcription used when no real STT backend is wired in (keeps the speech pipeline testable without external services).

### `app/llm/tts_provider.py` (Phase 8)
- **`TTSProvider(Protocol)`** — **`synthesize(self, text: str) -> bytes`**.
- **`class PassthroughTTSProvider`** — *"Minimal provider boundary for environments without a real TTS backend."*
  - **`__init__(self, provider_name, enabled) -> None`**
  - **`synthesize(self, text) -> bytes`** — placeholder audio synthesis mirroring the STT passthrough pattern.

---

## `app/youtube/` — Educational Resource Discovery

### `app/youtube/__init__.py`
Package marker (no content).

### `app/youtube/provider.py`
- **`YouTubeProvider(Protocol)`** — **`search_educational_videos(self, query, language, max_results) -> list[dict]`**, the contract implemented by concrete providers.

### `app/youtube/data_api_provider.py`
- **`class YouTubeDataAPIProvider`**
  - **`__init__(self, settings) -> None`** — stores API key/base URL/timeout from settings.
  - **`search_educational_videos(self, query, language, max_results) -> list[dict]`** — calls the real YouTube Data API v3 `search` endpoint (via `httpx`) and returns normalized candidate video metadata (id, title, description, channel, etc.) for downstream ranking.

---

## `app/knowledge/` — RAG Ingestion & Vector Search

### `app/knowledge/__init__.py`
Package marker (no content).

### `app/knowledge/transcript_provider.py`
- **`TranscriptProvider(Protocol)`** — **`get_transcript_for_resource(self, resource_id, resource) -> str`**, the contract for pulling raw transcript text from any resource type.

### `app/knowledge/youtube_transcript_provider.py`
- **`class YouTubeTranscriptProvider`**
  - **`__init__(self, languages) -> None`** — preferred caption languages.
  - **`get_transcript_for_resource(self, resource_id, resource) -> str`** — fetches captions via `youtube-transcript-api` for a YouTube resource and returns the concatenated transcript text.

### `app/knowledge/embedding_provider.py`
- **`EmbeddingProvider(Protocol)`** — `model_name` field; **`embed_text(self, text) -> list[float]`**; **`embed_documents(self, texts) -> list[list[float]]`**.
- **`class SentenceTransformerEmbeddingProvider`**
  - **`__init__(self, model_name) -> None`** — loads a `sentence-transformers` model.
  - **`embed_text(self, text) -> list[float]`** — single-text embedding.
  - **`embed_documents(self, texts) -> list[list[float]]`** — batch embedding for chunk sets.

### `app/knowledge/faiss_index_store.py`
- **`class FaissIndexStore`**
  - **`__init__(self, base_dir)`** — sets the directory where per-resource FAISS indexes are persisted.
  - **`build_index(self, resource_id, embeddings, chunk_ids) -> dict`** — builds a FAISS index from chunk embeddings, writes it to disk, and returns index metadata (path, chunk count, etc.) for the `KnowledgeIndex` DB record.
  - **`search(self, resource_id, query_vector, top_k) -> list[tuple[str, float]]`** — loads the resource's index and returns the top-k `(chunk_id, similarity_score)` matches for a query embedding.

---

## `app/services/` — Business Logic / Orchestration

### `app/services/__init__.py`
Package marker. Docstring: *"Application services."*

### `app/services/errors.py`
Shared service-level exception hierarchy, all rooted at `AssistantError(RuntimeError)` — *"Base class for assistant workflow failures."* Subclasses (each raised by a specific failure mode):
- `LearningSessionNotFoundError` — unknown session referenced by a text question.
- `LearnerNotFoundError` — missing learner record.
- `LearningSessionOwnershipError` — session doesn't belong to the targeted learner.
- `RuntimeConfigurationNotFoundError` — requested runtime configuration doesn't exist.
- `ResourceNotFoundError` — requested educational resource not found.
- `YouTubeProviderConfigurationError` — YouTube config missing/invalid.
- `YouTubeProviderError` — YouTube provider failed to return usable results.
- `NoRecommendationsFoundError` — no usable recommendations available.
- `KnowledgePreparationError` — transcript preparation/indexing failed.
- `KnowledgeRetrievalError` — knowledge retrieval failed.
- `AssistantProviderUnavailableError(message, interaction_id)` — raised after an LLM provider failure has been sanitized and the failed interaction recorded (carries the `interaction_id` so callers can look up what was persisted).
- `STTProviderError` / `TTSProviderError` — speech transcription/synthesis failures.

### `app/services/assistant_service.py` (Phase 2, 5, 6, 7 — the orchestration core)
- **`KnowledgeRetrievalServiceProtocol`** — structural type: `retrieve_context(self, resource_id, query, top_k)`.
- **`class AssistantResponse`** (dataclass-like) — `interaction_id`, `session_id`, `answer`, `response_latency_ms`, `model_provider`, `model_name`, `runtime_configuration_id`.
- **`class AssistantService`**
  - **`__init__(self, repository, llm_provider, settings, personalization_service, knowledge_retrieval_service, retrieval_quality_service) -> None`** — holds every collaborator needed to answer a question end-to-end.
  - **`answer_text_question(self, session_id, question, runtime_configuration_id, resource_id) -> AssistantResponse`** — the main pipeline: resolve session/runtime config → build personalization context → optionally retrieve RAG context (gated by CRAG quality check) → build prompt → call the LLM provider → measure latency → persist the `Interaction` (success or sanitized failure) → return `AssistantResponse`.
  - **`_resolve_runtime_configuration(self, runtime_configuration_id) -> RuntimeConfiguration`** — loads a specific config snapshot or falls back to the default named configuration, creating one if missing.
  - **`_default_runtime_configuration_data(self) -> dict`** — the default config payload (model id, temperature, etc.) used when no explicit config is supplied.
  - **`_build_personalization_context(self, learner_id)`** — pulls the learner's `PersonalizationContext` via `LearnerPersonalizationService`.
  - **`_build_llm_request(self, question, runtime_configuration, personalization_context, retrieval_context) -> LLMGenerateRequest`** — assembles the final prompt/system-prompt/model params sent to the LLM provider.
  - **`_retrieve_context_for_question(self, question, resource_id) -> str | None`** — calls the knowledge retrieval service for a resource, runs the CRAG `RetrievalQualityService` gate, and returns joined chunk text only if retrieval is judged adequate (else `None`, triggering a fallback/no-RAG prompt).
  - **`_build_system_prompt(self, personalization_context, retrieval_context) -> str`** — composes the system prompt, weaving in personalization (language, competency, interests, preferences) and any retrieved knowledge context.
  - **`_elapsed_ms(started_at: float) -> int`** — latency helper (ms since `started_at`).

### `app/services/crag_service.py` (Phase 7 — Corrective RAG quality gate)
- **`class RetrievalQualityDecision`** — `is_adequate: bool`, `reason: str`, `top_score: float | None`.
- **`class RetrievalQualityService`**
  - **`__init__(self, min_score) -> None`** — stores the configured `crag_quality_min_score` threshold.
  - **`evaluate(self, question, hits) -> RetrievalQualityDecision`** — inspects retrieved chunk scores and decides whether the retrieval is good enough to ground an answer, or whether the assistant should fall back to a non-RAG (or "I don't have enough context") response.

### `app/services/knowledge_preparation_service.py` (Phase 5 — ingestion pipeline)
- **`class PreparedKnowledgeDocument`** — `resource_id`, `document_id`, `chunk_count`, `status`.
- **`class KnowledgePreparationService`**
  - **`__init__(self, repository, transcript_provider, embedding_provider, vector_index_store, settings) -> None`**.
  - **`prepare_resource(self, resource_id) -> PreparedKnowledgeDocument`** — full pipeline: fetch resource → pull transcript → preprocess/clean text → chunk it → embed chunks → build/persist a FAISS index → persist `KnowledgeDocument`/`KnowledgeChunk`/`KnowledgeIndex` DB rows → return summary.
  - **`_preprocess_transcript(text) -> str`** — cleans raw transcript text (whitespace/formatting normalization) before chunking.
  - **`_chunk_text(text, chunk_size, overlap) -> list[str]`** — splits transcript text into overlapping chunks per `knowledge_chunk_size`/`knowledge_chunk_overlap` settings.

### `app/services/knowledge_retrieval_service.py` (Phase 5, 6 — query-time retrieval)
- **`class KnowledgeRetrievalService`**
  - **`__init__(self, repository, embedding_provider, vector_index_store, settings) -> None`**.
  - **`retrieve_context(self, resource_id, query, top_k) -> list[dict]`** — embeds the query, searches the resource's FAISS index, loads matching `KnowledgeChunk` rows, and returns ranked context dicts (content + score) for both the `/knowledge/retrieve` endpoint and the assistant's RAG step.

### `app/services/personalization_service.py` (Phase 3)
- **`class LearnerPersonalizationService`**
  - **`__init__(self, repository: LearningTraceRepository) -> None`**.
  - **`get_learner_personalization(self, learner_id) -> LearnerPersonalizationProfile | None`** — fetches stored preferences.
  - **`update_learner_personalization(self, learner_id, data) -> LearnerPersonalizationProfile`** — updates language/competency/interests/learning-preferences.
  - **`get_context_for_learner(self, learner_id) -> PersonalizationContext`** — builds the aggregated context (including per-topic competencies) injected into assistant prompts.
  - **`record_assessment_result(self, learner_id, data) -> AssessmentResultResponse`** — persists a new topic assessment result.

### `app/services/resource_recommendation_service.py` (Phase 4)
- **`class ResourceRecommendationService`**
  - **`__init__(self, repository, youtube_provider) -> None`**.
  - **`get_recommendations_for_session(self, session_id, request) -> ResourceRecommendationResponse`** — resolves the session's learner, fetches YouTube candidates, ranks them, and returns the response payload.
  - **`select_resource_for_session(self, session_id, resource_id) -> ResourceSelectionResponse`** — marks a chosen resource as active/selected for the session.
  - **`_build_personalization_context(self, learner_id) -> PersonalizationContext`** — pulls learner context used in ranking.
  - **`_rank_candidates(self, candidates, topic, personalization_context) -> list[ResourceRecommendationItem]`** — scores/orders YouTube candidates by relevance to topic + learner language/interests, producing a `ranking_score` and `relevance_reason` per item.
  - **`_persist_candidates(self, candidates, topic) -> None`** — get-or-creates `EducationalResource` DB rows for candidates so they can be referenced later (e.g. for knowledge preparation).
  - **`_normalize_language(language) -> str`** — normalizes/defaults the language code used in filtering/ranking.

### `app/services/speech_service.py` (Phase 8)
- **`class SpeechResponse`** — `interaction_id`, `session_id`, `transcript`, `answer`, `audio_base64`, `response_latency_ms`, `model_provider`, `model_name`, `runtime_configuration_id`; **`audio_bytes(self) -> bytes`** decodes the base64 audio back to raw bytes.
- **`class SpeechService`**
  - **`__init__(self, assistant_service, stt_provider, tts_provider, repository=None) -> None`** — in the current code, `repository` is optional and not used directly by the audio-question flow.
  - **`answer_audio_question(self, session_id, audio_bytes, mime_type, runtime_configuration_id, resource_id) -> SpeechResponse`** — orchestrates the full speech flow: STT transcribe → delegate to `AssistantService.answer_text_question` → TTS synthesize the answer → return combined transcript/answer/audio response, converting provider failures into `STTProviderError`/`TTSProviderError`.

### `app/services/monitoring_service.py` (Phase 9 — QoE)
- **`class MonitoringObservation`** — `metric_name`, `metric_category`, `metric_value`, `unit`, `observation_data`.
- **`class QoEScore`** — `interaction_id`, `session_id`, `score`, `quality_label`, `latency_ms`, `model_provider`, `model_name`, `details`.
- **`class MonitoringService`**
  - **`__init__(self, repository: LearningTraceRepository) -> None`**.
  - **`record_interaction_observations(self, interaction) -> list[object]`** — derives raw runtime metrics from an interaction (e.g. latency) and persists them as `RuntimeMetric` rows.
  - **`evaluate_qoe(self, interaction) -> QoEScore`** — computes a composite QoE score/quality label from the interaction's metrics and persists a `QoEScoreRecord`.
  - **`_quality_label(score) -> str`** — maps a numeric score to a human-readable quality bucket (e.g. good/fair/poor).
  - **`_build_observations(interaction) -> list[MonitoringObservation]`** — internal helper building the list of metric observations to record.

### `app/services/experience_service.py` (Phase 10 — experience memory)
- **`class ExperienceService`**
  - **`__init__(self, repository: LearningTraceRepository) -> None`**.
  - **`create_experience_for_interaction(self, interaction_id) -> Experience`** — assembles a full experience record for a completed interaction: state snapshot (personalization/session context), action snapshot (what the assistant did), configuration snapshot (runtime config used), QoE outcome, and a computed reward score/details — the data substrate intended for future self-adaptive policy learning.
  - **`get_experiences_for_session(self, session_id) -> list[Experience]`** — lists all experience records for a session.
  - **`get_experience_for_interaction(self, interaction_id) -> Experience | None`** — looks up the experience tied to one interaction.

### `app/services/learning_trace_service.py` (cross-phase convenience façade)
- **`class LearningTraceService`**
  - **`__init__(self, repository: LearningTraceRepository) -> None`**.
  - **`_coerce_model(self, data, model_type)`** — internal helper normalizing dict/pydantic input into the expected model type before delegating to the repository.
  - **`create_learner(self, data) -> LearnerProfile`**, **`start_session(self, data) -> LearningSession`**, **`create_resource(self, data) -> EducationalResource`**, **`create_runtime_configuration_snapshot(self, name, description, configuration_data) -> RuntimeConfiguration`**, **`record_interaction(self, data) -> Interaction`** — thin, testable wrappers around the repository for the core learning-trace entities (used by tests/other services and as a simpler façade over raw repository calls).

---

## `app/db/` — Persistence Layer

### `app/db/__init__.py`
Package marker. Docstring: *"Database package."*

### `app/db/base.py`
- **`enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None`** — SQLAlchemy event hook that turns on `PRAGMA foreign_keys=ON` for every new SQLite connection (SQLite disables FK enforcement by default).
- **`utc_now() -> datetime`** — timezone-aware UTC timestamp helper used as the default for `created_at`/`updated_at`.
- **`class Base(DeclarativeBase)`** — the shared SQLAlchemy declarative base all ORM models inherit from.
- **`class TimestampMixin`** — mixin adding `created_at: Mapped[datetime]` and `updated_at: Mapped[datetime]` to any model that includes it.

### `app/db/errors.py`
- **`class ImmutableRuntimeConfigurationError(RuntimeError)`** — *"Raised when a referenced runtime configuration snapshot is modified."* Enforces the "runtime configs are immutable once referenced" invariant (see `runtime_configuration.py` below).

### `app/db/init_db.py`
- **`create_tables(engine) -> None`** — creates all ORM tables against the given engine (used at startup / in tests for a fresh SQLite DB).

### `app/db/session.py`
Session/engine setup (session-maker bound to the resolved database URL) used by `get_db_session` and tests. No top-level functions extracted (module builds the engine/sessionmaker at import time).

### `app/db/models/__init__.py`
Package marker; re-exports the ORM models for convenient importing elsewhere.

### `app/db/models/learner.py` (Phase 1, 3)
- **`class LearnerProfile(TimestampMixin, Base)`** — `id`, `display_name`, `profile_data` (JSON), `preferred_language`, `competency_level`, `interests` (list[str]), `learning_preferences` (JSON); relationships to `sessions`, `assessment_results`, `experiences`.

### `app/db/models/learning_session.py` (Phase 1)
- **`class LearningSession(TimestampMixin, Base)`** — `id`, `learner_id`, `started_at`, `ended_at`, `status`; relationships to `learner`, `interactions`, `runtime_metrics`, `qoe_scores`, `experiences`.

### `app/db/models/interaction.py` (Phase 2)
- **`class Interaction(TimestampMixin, Base)`** — `id`, `session_id`, `runtime_configuration_id`, `educational_resource_id`, `user_input`, `assistant_output`, `occurred_at`, `response_latency_ms`, `model_provider`, `model_name`, `status`, `error_message`; relationships to `session`, `runtime_configuration`, `educational_resource`, `runtime_metrics`, `qoe_scores`, `experiences`. This is the central row every question/answer exchange produces.

### `app/db/models/assessment_result.py` (Phase 3)
- **`class AssessmentResult(TimestampMixin, Base)`** — `id`, `learner_id`, `topic`, `competency_level`, `score`, `assessment_data` (JSON); relationship to `learner`.

### `app/db/models/educational_resource.py` (Phase 4)
- **`class EducationalResource(TimestampMixin, Base)`** — `id`, `source`, `external_resource_id`, `title`, `url`, `topic`, `resource_metadata` (JSON); relationships to `interactions`, `knowledge_documents`, `experiences`.

### `app/db/models/knowledge_document.py` (Phase 5)
- **`class KnowledgeDocument(TimestampMixin, Base)`** — `id`, `resource_id`, `title`, `source`, `transcript_text`, `status`, `document_metadata` (JSON); relationships to `resource`, `chunks`, `indices`.

### `app/db/models/knowledge_chunk.py` (Phase 5)
- **`class KnowledgeChunk(TimestampMixin, Base)`** — `id`, `document_id`, `chunk_index`, `content`, `chunk_metadata` (JSON), `status`; relationship to `document`.

### `app/db/models/knowledge_index.py` (Phase 5)
- **`class KnowledgeIndex(TimestampMixin, Base)`** — `id`, `document_id`, `index_name`, `index_path`, `embedding_model`, `chunk_count`, `status`, `index_metadata` (JSON); relationship to `document`. Tracks the on-disk FAISS index tied to a document.

### `app/db/models/runtime_configuration.py` (Phase 1–10)
- **`class RuntimeConfiguration(TimestampMixin, Base)`** — `id`, `name`, `version`, `description`, `configuration_data` (JSON); relationships to `interactions`, `experiences`. Versioned, immutable snapshots of "which model/params/routing mode was used."
- **`prevent_referenced_runtime_configuration_updates(mapper, connection, target) -> None`** — SQLAlchemy event hook that raises `ImmutableRuntimeConfigurationError` if code tries to mutate a `RuntimeConfiguration` row that is already referenced by an interaction/experience — enforces immutability of historical config snapshots (important for reproducible QoE/experience analysis).

### `app/db/models/runtime_metric.py` (Phase 9)
- **`class RuntimeMetric(TimestampMixin, Base)`** — `id`, `interaction_id`, `session_id`, `metric_name`, `metric_category`, `metric_value`, `unit`, `observed_at`, `observation_data` (JSON); relationships to `interaction`, `session`. Raw metrics feeding QoE computation.

### `app/db/models/qoe_score.py` (Phase 9)
- **`class QoEScoreRecord(TimestampMixin, Base)`** — `id`, `interaction_id`, `session_id`, `score`, `quality_label`, `latency_ms`, `model_provider`, `model_name`, `observed_at`; relationships to `interaction`, `session`, `experience`.

### `app/db/models/experience.py` (Phase 10)
- **`class Experience(TimestampMixin, Base)`** — `id`, `learner_id`, `session_id`, `interaction_id`, `runtime_configuration_id`, `qoe_score_id`, `resource_id`, `topic`, `state_snapshot` (JSON), `action_snapshot` (JSON), `configuration_snapshot` (JSON), `qoe_outcome` (JSON), `reward_score`, `reward_details` (JSON), `outcome_label`, `performance_summary` (JSON); relationships to `learner`, `session`, `interaction`, `runtime_configuration`, `qoe_score`, `resource`. The full "state/action/config → QoE → reward" record used for future adaptive-policy learning.

### `app/db/repositories/__init__.py`
Package marker (no content).

### `app/db/repositories/learning_trace.py`
The single data-access layer wrapping SQLAlchemy CRUD for (almost) every model above.
- **`class LearningTraceRepository`**
  - **`__init__(self, session: Session) -> None`**
  - **Learner:** `create_learner(data) -> LearnerProfile`, `get_learner(learner_id) -> LearnerProfile | None`, `update_learner_personalization(learner_id, data) -> LearnerProfile | None`.
  - **Assessments:** `create_assessment_result(learner_id, data) -> AssessmentResult`, `list_assessment_results(learner_id) -> list[AssessmentResult]`.
  - **Sessions:** `create_learning_session(data) -> LearningSession`, `get_learning_session(session_id) -> LearningSession | None`.
  - **Resources:** `create_educational_resource(data) -> EducationalResource`, `get_educational_resource_by_source_and_external_id(source, external_resource_id) -> EducationalResource | None`, `get_or_create_educational_resource(source, external_resource_id, title, url, topic, metadata) -> EducationalResource`, `get_educational_resource(resource_id) -> EducationalResource | None`.
  - **Runtime configuration:** `create_runtime_configuration(data) -> RuntimeConfiguration`, `get_runtime_configuration(id) -> RuntimeConfiguration | None`, `get_latest_runtime_configuration_by_name(name) -> RuntimeConfiguration | None`, `create_runtime_configuration_version(name, description, configuration_data) -> RuntimeConfiguration`, `get_or_create_runtime_configuration_snapshot(name, description, configuration_data) -> RuntimeConfiguration`, `_next_runtime_configuration_version(name) -> int` (internal version-number helper).
  - **Interactions:** `create_interaction(data) -> Interaction`, `get_interaction(interaction_id) -> Interaction | None`, `get_interaction_with_trace(interaction_id) -> Interaction | None` (eager-loads related metrics/QoE/experience for full-trace views).
  - **Monitoring/QoE:** `create_runtime_metric(interaction_id, session_id, metric_name, metric_category, metric_value, unit, observation_data) -> RuntimeMetric`, `create_qoe_score(interaction_id, session_id, score, quality_label, latency_ms, model_provider, model_name, details) -> QoEScoreRecord`, `get_qoe_score_for_interaction(interaction_id) -> QoEScoreRecord | None`.
  - **Experience memory:** `create_experience(learner_id, session_id, interaction_id, runtime_configuration_id, qoe_score_id, resource_id, topic, state_snapshot, action_snapshot, configuration_snapshot, qoe_outcome, reward_score, reward_details, outcome_label, performance_summary) -> Experience`, `get_experience_for_interaction(interaction_id) -> Experience | None`, `get_experiences_for_session(session_id) -> list[Experience]`.
  - **Knowledge/RAG:** `create_knowledge_document(resource_id, title, source, transcript_text, metadata) -> KnowledgeDocument`, `get_knowledge_document_for_resource(resource_id) -> KnowledgeDocument | None`, `create_knowledge_chunk(document_id, chunk_index, content, metadata) -> KnowledgeChunk`, `attach_document_chunks(document_id, chunk_ids) -> None`, `get_knowledge_chunk(chunk_id) -> KnowledgeChunk | None`, `create_knowledge_index(document_id, index_name, index_path, embedding_model, chunk_count, metadata) -> KnowledgeIndex`, `update_knowledge_document_status(document_id, status) -> KnowledgeDocument | None`.

---

## `tests/` — Test Suite

### `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`
Package markers (docstrings: "Test package.", "Unit tests.", "Integration tests.").

### Unit Tests (`tests/unit/`)

- **`test_config.py`** — `test_resolves_local_edge_storage_paths_from_environment(monkeypatch)`, `test_database_url_overrides_database_path()`: validate `Settings` path resolution and the `database_url` override precedence.
- **`test_assistant_service.py`** — `FakeLLMProvider` (`__init__(fail)`, `generate(request)`), fixtures `db_session()`, `create_learning_session(db_session)`; tests: `test_assistant_records_successful_text_question`, `test_assistant_uses_personalization_context_in_prompt`, `test_learner_personalization_service_builds_context`, `test_assistant_rejects_unknown_session`, `test_assistant_records_failed_interaction_for_provider_error` — cover the main `AssistantService.answer_text_question` happy/error paths and personalization prompt injection.
- **`test_experience_service.py`** — `db_session()` fixture; `test_experience_service_creates_persistent_experience_record` — verifies `ExperienceService` correctly builds/persists a full experience row.
- **`test_huggingface_provider.py`** — `FakeResponse` (`__init__`, `raise_for_status`, `json`), `make_request() -> LLMGenerateRequest`; tests: `test_huggingface_provider_sends_chat_completion_request`, `test_huggingface_provider_sanitizes_http_errors`, `test_huggingface_provider_rejects_malformed_response` — validate the HF provider's request shape and error handling (monkeypatched HTTP).
- **`test_knowledge_preparation_service.py`** — fakes `StaticTranscriptProvider`, `StaticEmbeddingProvider`, `StaticVectorIndexStore`; tests: `test_knowledge_preparation_service_builds_chunks_and_index`, `test_assistant_uses_retrieved_context_in_prompt`, `test_assistant_preserves_fallback_without_retrieval_context` — covers chunking/indexing and that the assistant correctly folds retrieved context into prompts (or falls back when there is none).
- **`test_monitoring_service.py`** — `db_session()` fixture; `test_monitoring_service_records_and_scores_qoe` — validates metric recording + QoE scoring end-to-end against a DB.
- **`test_phase7_routing_and_crag.py`** — `FakeCloudProvider`, `FakeLocalProvider`; tests: `test_crag_quality_service_flags_inadequate_retrieval`, `test_assistant_uses_fallback_prompt_when_retrieval_is_inadequate`, `test_hybrid_provider_routes_to_cloud_when_local_unavailable`, `test_hybrid_provider_uses_cloud_fallback_when_local_provider_fails` — covers the CRAG quality gate and the hybrid edge/cloud routing/fallback logic.
- **`test_resource_recommendation_service.py`** — `FakeYouTubeProvider`; tests: `test_recommendation_service_ranks_by_language_and_interest`, `test_recommendation_service_raises_when_no_results_available` — validates ranking logic and the `NoRecommendationsFoundError` path.
- **`test_speech_service.py`** — `FakeSTTProvider`, `FakeTTSProvider`, `FakeLLMProvider`; fixtures `db_session()`, `create_learning_session(db_session)`; tests: `test_stt_provider_success_and_failure`, `test_tts_provider_success_and_failure`, `test_speech_service_orchestrates_audio_to_assistant_to_tts`, `test_speech_service_propagates_stt_and_tts_errors` — covers the full speech pipeline plus provider error propagation.

### Integration Tests (`tests/integration/`)

- **`test_text_question_api.py`** — `FakeLLMProvider`; `test_text_question_endpoint_returns_answer_and_persists_interaction` — exercises `POST` text-question route through the real FastAPI app + DB.
- **`test_speech_api.py`** — `FakeSTTProvider`, `FakeTTSProvider`, `FakeLLMProvider`; `test_speech_endpoint_returns_answer_and_audio` — exercises the speech route end-to-end.
- **`test_personalization_api.py`** — `FakeLLMProvider`; `test_personalization_routes_and_isolation_work` — verifies personalization endpoints and that learner data stays isolated per learner.
- **`test_learning_trace_persistence.py`** — `db_session()` fixture; tests: `test_persists_full_learning_trace_without_interaction_learner_duplication`, `test_invalid_session_learner_foreign_key_is_rejected`, `test_runtime_configuration_updates_create_new_versions_for_history`, `test_referenced_runtime_configuration_snapshot_cannot_be_modified` — validates FK integrity, versioning behavior, and the immutability guard on referenced runtime configurations (see `prevent_referenced_runtime_configuration_updates`).

---

## Notes

- Function-level details above were extracted directly from the source via Python's `ast` module (class/method/function names, parameters, and return annotations), then annotated with purpose based on naming, call graph, and cross-referencing `architecture.md`'s phase roadmap — so this reflects the actual current code, not just design docs.
- Most methods have no inline docstrings in the source; the one-line purposes given here are inferred from signatures, call sites, and surrounding context rather than quoted from comments (the few `MODULE DOC:` / class docstrings that do exist are called out explicitly, e.g. `AssistantProviderUnavailableError`, `PassthroughSTTProvider`).
- Protocol classes (`LLMProvider`, `STTProvider`, `TTSProvider`, `EmbeddingProvider`, `TranscriptProvider`, `YouTubeProvider`, `KnowledgeRetrievalServiceProtocol`) define structural interfaces with no implementation of their own — they exist purely so services can depend on an abstraction rather than a concrete provider (enables the fakes used throughout `tests/`).
- The "self-adaptive" aspect of the architecture is centered on `app/db/models/experience.py` + `app/services/experience_service.py`: every interaction can be converted into a state/action/config/QoE/reward record, which is the data phase-10 is laying the groundwork for (an eventual policy that picks runtime configuration based on past reward).
- This manifest can be regenerated any time the source changes by re-running an AST-based extraction pass over `app/` and `tests/`.
