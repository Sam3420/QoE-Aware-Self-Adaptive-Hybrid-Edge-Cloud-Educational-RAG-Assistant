from __future__ import annotations

from typing import Any

from app.db.models import Experience, Interaction, LearningSession
from app.db.repositories import LearningTraceRepository


class ExperienceService:
    def __init__(self, repository: LearningTraceRepository) -> None:
        self.repository = repository

    def create_experience_for_interaction(self, interaction_id: str) -> Experience:
        interaction = self.repository.get_interaction(interaction_id)
        if interaction is None:
            raise ValueError(f"Interaction not found: {interaction_id}")

        qoe = self.repository.get_qoe_score_for_interaction(interaction_id)
        state_snapshot = {
            "session_id": interaction.session_id,
            "learner_id": interaction.session.learner_id,
            "resource_id": interaction.educational_resource_id,
            "topic": interaction.educational_resource.topic if interaction.educational_resource else None,
            "user_input": interaction.user_input,
        }
        action_snapshot = {
            "action_type": "text_question",
            "runtime_configuration_id": interaction.runtime_configuration_id,
            "model_provider": interaction.model_provider,
            "model_name": interaction.model_name,
            "response_latency_ms": interaction.response_latency_ms,
        }
        configuration_snapshot = {
            "name": interaction.runtime_configuration.name,
            "version": interaction.runtime_configuration.version,
            "configuration_data": interaction.runtime_configuration.configuration_data,
        }
        qoe_outcome = {
            "interaction_id": interaction.id,
            "score": qoe.score if qoe else 0.0,
            "quality_label": qoe.quality_label if qoe else "poor",
            "latency_ms": qoe.latency_ms if qoe else interaction.response_latency_ms,
        }
        reward_score = float(qoe.score if qoe else 0.0)
        reward_details = {
            "reward_basis": "qoe_score",
            "qoe_score": reward_score,
            "sre_fallback": (
                "SRE parts (CPU, MEM, EN, NE) are not currently collected in this runtime, "
                "so reward is recorded from the observed QoE score only."
            ),
        }
        outcome_label = qoe.quality_label if qoe else "poor"
        performance_summary = {
            "status": interaction.status,
            "answer_length": len(interaction.assistant_output or ""),
            "resource_used": interaction.educational_resource_id is not None,
        }

        existing = self.repository.get_experience_for_interaction(interaction_id)
        if existing is not None:
            return existing

        return self.repository.create_experience(
            learner_id=interaction.session.learner_id,
            session_id=interaction.session_id,
            interaction_id=interaction.id,
            runtime_configuration_id=interaction.runtime_configuration_id,
            qoe_score_id=qoe.id if qoe else None,
            resource_id=interaction.educational_resource_id,
            topic=interaction.educational_resource.topic if interaction.educational_resource else None,
            state_snapshot=state_snapshot,
            action_snapshot=action_snapshot,
            configuration_snapshot=configuration_snapshot,
            qoe_outcome=qoe_outcome,
            reward_score=reward_score,
            reward_details=reward_details,
            outcome_label=outcome_label,
            performance_summary=performance_summary,
        )

    def get_experiences_for_session(self, session_id: str) -> list[Experience]:
        return self.repository.get_experiences_for_session(session_id)

    def get_experience_for_interaction(self, interaction_id: str) -> Experience | None:
        return self.repository.get_experience_for_interaction(interaction_id)
