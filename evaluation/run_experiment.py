from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.db.base import Base
from app.db.repositories import LearningTraceRepository
from app.domain.models import LearnerProfileCreate, LearningSessionCreate
from app.knowledge.embedding_provider import SentenceTransformerEmbeddingProvider
from app.knowledge.faiss_index_store import FaissIndexStore
from app.knowledge.youtube_transcript_provider import YouTubeTranscriptProvider
from app.llm.huggingface_provider import HuggingFaceInferenceProvider
from app.llm.hybrid_provider import HybridLLMProvider, LocalLLMProvider
from app.services.adaptive_intelligence_service import AdaptiveIntelligenceService
from app.services.assistant_service import AssistantService
from app.services.crag_service import RetrievalQualityService
from app.services.experience_service import ExperienceService
from app.services.knowledge_preparation_service import KnowledgePreparationService
from app.services.knowledge_retrieval_service import KnowledgeRetrievalService
from app.services.learning_trace_service import LearningTraceService
from app.services.monitoring_service import MonitoringService
from app.services.personalization_service import LearnerPersonalizationService

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT / "evaluation"
FIGURES_DIR = EVALUATION_DIR / "figures"
QUESTIONS_FILE = EVALUATION_DIR / "questions.json"
RESULTS_FILE = EVALUATION_DIR / "results.csv"
JSON_RESULTS_FILE = EVALUATION_DIR / "results.json"
AGGREGATE_FILE = EVALUATION_DIR / "aggregate.csv"

EVALUATION_RESOURCE = {
    "source": "youtube",
    "external_resource_id": "DSIdaTSG2Gg",
    "title": "An Animated Introduction to Social Science",
    "url": "https://www.youtube.com/watch?v=DSIdaTSG2Gg",
    "topic": "social science",
}


def load_questions() -> list[dict[str, str]]:
    return json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))


def validate_runtime_requirements(settings: Settings) -> None:
    if not settings.hf_token or not settings.hf_token.get_secret_value():
        raise RuntimeError(
            "Hugging Face token is not configured. Set HF_TOKEN in .env to run the real cloud provider path."
        )

    try:
        import faiss  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "faiss is required for the real retrieval pipeline. Install the vector index dependency before running evaluation."
        ) from exc

    try:
        import sentence_transformers  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "sentence-transformers is required for the real embedding pipeline. Install the package before running evaluation."
        ) from exc


def initial_runtime_configuration(strategy: str, settings: Settings) -> dict[str, object]:
    provider_by_strategy = {
        "cloud_only": "cloud",
        "static_hybrid": "local",
        "adaptive_hybrid": "cloud",
    }
    return {
        "provider": provider_by_strategy.get(strategy, "cloud"),
        "model_id": settings.llm_model_id,
        "max_tokens": settings.llm_max_tokens,
        "temperature": settings.llm_temperature,
    }


def build_llm_provider(settings: Settings) -> HybridLLMProvider:
    token = settings.hf_token.get_secret_value() if settings.hf_token else ""
    cloud_provider = HuggingFaceInferenceProvider(
        token=token,
        inference_url=settings.hf_inference_url,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    local_provider = LocalLLMProvider(model_id=settings.local_llm_model_id)
    return HybridLLMProvider(
        cloud_provider=cloud_provider,
        local_provider=local_provider,
        settings=settings,
    )


def build_knowledge_retrieval_service(
    repository: LearningTraceRepository,
    settings: Settings,
) -> KnowledgeRetrievalService:
    return KnowledgeRetrievalService(
        repository=repository,
        embedding_provider=SentenceTransformerEmbeddingProvider(
            model_name=settings.knowledge_embedding_model,
        ),
        vector_index_store=FaissIndexStore(base_dir=settings.faiss_storage_path),
        settings=settings,
    )


def ensure_prepared_resource(
    repository: LearningTraceRepository,
    settings: Settings,
) -> str:
    resource = repository.get_or_create_educational_resource(
        source=EVALUATION_RESOURCE["source"],
        external_resource_id=EVALUATION_RESOURCE["external_resource_id"],
        title=EVALUATION_RESOURCE["title"],
        url=EVALUATION_RESOURCE["url"],
        topic=EVALUATION_RESOURCE["topic"],
        metadata={"evaluation": True},
    )

    if repository.get_knowledge_document_for_resource(resource.id) is None:
        preparation_service = KnowledgePreparationService(
            repository=repository,
            transcript_provider=YouTubeTranscriptProvider(
                languages=[settings.youtube_default_language],
            ),
            embedding_provider=SentenceTransformerEmbeddingProvider(
                model_name=settings.knowledge_embedding_model,
            ),
            vector_index_store=FaissIndexStore(base_dir=settings.faiss_storage_path),
            settings=settings,
        )
        preparation_service.prepare_resource(resource.id)
        repository.session.commit()

    return resource.id


def build_assistant(
    settings: Settings,
    repository: LearningTraceRepository,
    llm_provider,
    knowledge_retrieval_service: KnowledgeRetrievalService,
) -> AssistantService:
    return AssistantService(
        repository=repository,
        llm_provider=llm_provider,
        settings=settings,
        personalization_service=LearnerPersonalizationService(repository),
        knowledge_retrieval_service=knowledge_retrieval_service,
        retrieval_quality_service=RetrievalQualityService(min_score=settings.crag_quality_min_score),
        monitoring_service=MonitoringService(repository),
        experience_service=ExperienceService(repository),
        adaptive_intelligence_service=AdaptiveIntelligenceService(repository),
    )


def build_aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregates: list[dict[str, object]] = []
    for configuration in sorted({row["configuration"] for row in rows}):
        subset = [row for row in rows if row["configuration"] == configuration]
        aggregates.append(
            {
                "configuration": configuration,
                "mean_latency": round(mean([float(row["latency_ms"]) for row in subset]), 2),
                "mean_qoe": round(mean([float(row["qoe_score"]) for row in subset]), 2),
                "success_rate": round(
                    sum(1 for row in subset if row["success"]) / len(subset),
                    4,
                ),
                "cloud_usage": sum(1 for row in subset if row["provider"] == "cloud"),
                "local_usage": sum(1 for row in subset if row["provider"] == "local"),
            }
        )
    return aggregates


def write_outputs(rows: list[dict[str, object]]) -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "question_id",
        "configuration",
        "latency_ms",
        "qoe_score",
        "quality_label",
        "provider",
        "adaptive_action",
        "success",
    ]

    with RESULTS_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with JSON_RESULTS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    with AGGREGATE_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["configuration", "mean_latency", "mean_qoe", "success_rate", "cloud_usage", "local_usage"],
        )
        writer.writeheader()
        writer.writerows(build_aggregate_rows(rows))


def save_graphs(rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required to generate evaluation charts.") from exc

    configurations = ["cloud_only", "static_hybrid", "adaptive_hybrid"]

    qoe_by_config = {config: [] for config in configurations}
    latency_by_config = {config: [] for config in configurations}
    provider_counts = {config: {"cloud": 0, "local": 0} for config in configurations}

    for row in rows:
        config = row["configuration"]
        if config not in qoe_by_config:
            continue
        qoe_by_config[config].append(float(row["qoe_score"]))
        latency_by_config[config].append(float(row["latency_ms"]))
        if row["provider"] in {"cloud", "local"}:
            provider_counts[config][row["provider"]] += 1

    qoe_means = [mean(qoe_by_config[config]) if qoe_by_config[config] else 0.0 for config in configurations]
    latency_means = [mean(latency_by_config[config]) if latency_by_config[config] else 0.0 for config in configurations]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(configurations, qoe_means, color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_title("QoE Comparison")
    ax.set_ylabel("Mean QoE")
    ax.set_xlabel("Configuration")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "qoe_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(configurations, latency_means, color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_title("Latency Comparison")
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_xlabel("Configuration")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "latency_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = list(range(len(configurations)))
    bar_width = 0.35
    ax.bar([x - bar_width / 2 for x in x_positions], [provider_counts[config]["cloud"] for config in configurations], width=bar_width, label="cloud")
    ax.bar([x + bar_width / 2 for x in x_positions], [provider_counts[config]["local"] for config in configurations], width=bar_width, label="local")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(configurations)
    ax.set_title("Provider Usage Comparison")
    ax.set_ylabel("Usage Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "provider_usage_comparison.png", dpi=150)
    plt.close(fig)


def run_configuration(strategy: str) -> list[dict[str, object]]:
    settings = Settings(
        database_url=f"sqlite:///{(ROOT / 'data' / f'evaluation_{strategy}.db').as_posix()}",
        local_llm_enabled=False,
        llm_model_id="meta-llama/Llama-3.1-8B-Instruct",
        default_runtime_configuration_name="huggingface-llama-text",
    )
    validate_runtime_requirements(settings)

    db_path = ROOT / "data" / f"evaluation_{strategy}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    engine = create_engine(settings.database_url, future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    with Session() as session:
        repository = LearningTraceRepository(session)
        trace_service = LearningTraceService(repository)
        learner = trace_service.create_learner(LearnerProfileCreate(display_name="Evaluation Learner"))
        session_record = trace_service.start_session(LearningSessionCreate(learner_id=learner.id))

        initial_runtime = repository.get_or_create_runtime_configuration_snapshot(
            name=settings.default_runtime_configuration_name,
            description=f"{strategy} initial runtime configuration",
            configuration_data=initial_runtime_configuration(strategy, settings),
        )

        resource_id = ensure_prepared_resource(repository, settings)
        llm_provider = build_llm_provider(settings)
        knowledge_retrieval_service = build_knowledge_retrieval_service(repository, settings)
        assistant = build_assistant(
            settings,
            repository,
            llm_provider,
            knowledge_retrieval_service,
        )

        results: list[dict[str, object]] = []

        for question in load_questions():
            current_runtime = repository.get_latest_runtime_configuration_by_name(
                settings.default_runtime_configuration_name,
            )
            if current_runtime is None:
                current_runtime = initial_runtime

            started_at = perf_counter()
            try:
                response = assistant.answer_text_question(
                    session_id=session_record.id,
                    question=question["question"],
                    runtime_configuration_id=current_runtime.id,
                    resource_id=resource_id,
                )
                latency_ms = max(0, int((perf_counter() - started_at) * 1000))
                qoe = repository.get_qoe_score_for_interaction(response.interaction_id)
                if qoe is None:
                    raise RuntimeError("QoE record was not persisted.")

                provider = response.model_provider
                latest_runtime = repository.get_latest_runtime_configuration_by_name(
                    settings.default_runtime_configuration_name,
                )
                adaptive_action = "none"
                if latest_runtime is not None and latest_runtime.id != current_runtime.id:
                    description = latest_runtime.description or ""
                    if description.startswith("Adaptive config update from "):
                        adaptive_action = description.split("from ", 1)[1]
                    else:
                        adaptive_action = latest_runtime.configuration_data.get("provider", "cloud")

                qoe_score = qoe.score
                quality_label = qoe.quality_label
                success = True
            except Exception:
                latency_ms = max(0, int((perf_counter() - started_at) * 1000))
                provider = "unavailable"
                qoe_score = 0.0
                quality_label = "failure"
                adaptive_action = "none"
                success = False

            results.append(
                {
                    "question_id": question["id"],
                    "configuration": strategy,
                    "latency_ms": latency_ms,
                    "qoe_score": qoe_score,
                    "quality_label": quality_label,
                    "provider": provider,
                    "adaptive_action": adaptive_action,
                    "success": success,
                }
            )

        session.commit()

    return results


if __name__ == "__main__":
    strategies = ["cloud_only", "static_hybrid", "adaptive_hybrid"]
    all_rows: list[dict[str, object]] = []
    for strategy in strategies:
        all_rows.extend(run_configuration(strategy))

    write_outputs(all_rows)
    save_graphs(all_rows)

    print(f"Wrote {RESULTS_FILE}")
    print(f"Wrote {JSON_RESULTS_FILE}")
    print(f"Wrote {AGGREGATE_FILE}")
    for figure in [
        FIGURES_DIR / "qoe_comparison.png",
        FIGURES_DIR / "latency_comparison.png",
        FIGURES_DIR / "provider_usage_comparison.png",
    ]:
        print(f"Wrote {figure}")
