from dataclasses import dataclass, field
from typing import Any

from app.db.models import Interaction
from app.db.repositories import LearningTraceRepository


@dataclass(frozen=True)
class MonitoringObservation:
    metric_name: str
    metric_category: str
    metric_value: float
    unit: str | None = None
    observation_data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QoEScore:
    interaction_id: str
    session_id: str
    score: float
    quality_label: str
    latency_ms: int | None = None
    model_provider: str | None = None
    model_name: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class MonitoringService:
    def __init__(self, repository: LearningTraceRepository) -> None:
        self.repository = repository

    def record_interaction_observations(self, interaction: Interaction) -> list[object]:
        observations = self._build_observations(interaction)
        persisted: list[object] = []
        for observation in observations:
            persisted.append(
                self.repository.create_runtime_metric(
                    interaction_id=interaction.id,
                    session_id=interaction.session_id,
                    metric_name=observation.metric_name,
                    metric_category=observation.metric_category,
                    metric_value=observation.metric_value,
                    unit=observation.unit,
                    observation_data=observation.observation_data,
                )
            )
        return persisted

    def evaluate_qoe(self, interaction: Interaction) -> QoEScore:
        latency_ms = interaction.response_latency_ms or 0
        answer_length = len(interaction.assistant_output or "")
        completion_score = 100.0 if interaction.status == "completed" else 25.0
        latency_score = max(0.0, 100.0 - (latency_ms / 10.0))
        content_score = min(100.0, max(0.0, (answer_length / 250.0) * 100.0))
        score = (completion_score * 0.55) + (latency_score * 0.30) + (content_score * 0.15)
        score = round(max(0.0, min(100.0, score)), 2)
        quality_label = self._quality_label(score)

        qoe_record = self.repository.get_qoe_score_for_interaction(interaction.id)
        if qoe_record is None:
            qoe_record = self.repository.create_qoe_score(
                interaction_id=interaction.id,
                session_id=interaction.session_id,
                score=score,
                quality_label=quality_label,
                latency_ms=latency_ms,
                model_provider=interaction.model_provider,
                model_name=interaction.model_name,
                details={
                    "completion_score": completion_score,
                    "latency_score": latency_score,
                    "content_score": content_score,
                },
            )
        else:
            qoe_record.score = score
            qoe_record.quality_label = quality_label
            qoe_record.latency_ms = latency_ms
            qoe_record.model_provider = interaction.model_provider
            qoe_record.model_name = interaction.model_name
            qoe_record.observed_at = qoe_record.observed_at
            self.repository.session.flush()

        return QoEScore(
            interaction_id=interaction.id,
            session_id=interaction.session_id,
            score=qoe_record.score,
            quality_label=qoe_record.quality_label,
            latency_ms=qoe_record.latency_ms,
            model_provider=qoe_record.model_provider,
            model_name=qoe_record.model_name,
            details={
                "completion_score": completion_score,
                "latency_score": latency_score,
                "content_score": content_score,
            },
        )

    @staticmethod
    def _quality_label(score: float) -> str:
        if score >= 85.0:
            return "excellent"
        if score >= 70.0:
            return "good"
        if score >= 50.0:
            return "fair"
        return "poor"

    @staticmethod
    def _build_observations(interaction: Interaction) -> list[MonitoringObservation]:
        status_value = 1.0 if interaction.status == "completed" else 0.0
        answer_length = len(interaction.assistant_output or "")
        return [
            MonitoringObservation(
                metric_name="response_latency_ms",
                metric_category="performance",
                metric_value=float(interaction.response_latency_ms or 0),
                unit="ms",
                observation_data={"status": interaction.status},
            ),
            MonitoringObservation(
                metric_name="interaction_status",
                metric_category="outcome",
                metric_value=status_value,
                unit="bool",
                observation_data={"status": interaction.status},
            ),
            MonitoringObservation(
                metric_name="answer_length_chars",
                metric_category="quality",
                metric_value=float(answer_length),
                unit="chars",
                observation_data={"status": interaction.status},
            ),
        ]
