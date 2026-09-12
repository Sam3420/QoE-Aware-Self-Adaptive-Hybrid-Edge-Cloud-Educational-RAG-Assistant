from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from app.db.models import Experience
from app.db.repositories import LearningTraceRepository


ALLOWED_ACTIONS = [
    "CHANGE_MODEL",
    "CHANGE_RETRIEVAL_K",
    "CHANGE_RESPONSE_DETAIL",
    "CHANGE_RESPONSE_STYLE",
    "CHANGE_LANGUAGE",
    "CHANGE_PROVIDER",
]


@dataclass(frozen=True)
class AdaptiveActionDecision:
    action: str | None
    reason: str
    suggested_runtime_configuration: dict[str, Any] | None = None
    confidence: float = 0.0


class AdaptiveIntelligenceService:
    def __init__(self, repository: LearningTraceRepository) -> None:
        self.repository = repository

    def get_allowed_actions(self) -> list[str]:
        return list(ALLOWED_ACTIONS)

    def select_action(
        self,
        *,
        learner_id: str | None,
        current_qoe_score: float | None = None,
        current_runtime_configuration: dict[str, Any] | None = None,
        latency_ms: int | None = None,
        retrieval_quality_is_adequate: bool | None = None,
        personalization_context: dict[str, Any] | None = None,
    ) -> AdaptiveActionDecision | None:
        current_runtime_configuration = current_runtime_configuration or {}
        provider = str(current_runtime_configuration.get("provider", "huggingface")).lower()
        recent_scores = self._recent_qoe_scores(learner_id)
        average_recent_score = (
            sum(recent_scores) / len(recent_scores) if recent_scores else None
        )

        latest_qoe_score = current_qoe_score
        if latest_qoe_score is None and recent_scores:
            latest_qoe_score = recent_scores[0]

        experience_based_decision = self._select_action_from_experiences(
            learner_id=learner_id,
            current_runtime_configuration=current_runtime_configuration,
            personalization_context=personalization_context,
        )
        if experience_based_decision is not None:
            return experience_based_decision

        if retrieval_quality_is_adequate is False:
            return AdaptiveActionDecision(
                action="CHANGE_RETRIEVAL_K",
                reason=(
                    "Retrieval quality was insufficient; increasing the retrieval depth is the safest deterministic action."
                ),
                suggested_runtime_configuration=self.transform_configuration(
                    action="CHANGE_RETRIEVAL_K",
                    current_runtime_configuration=current_runtime_configuration,
                    personalization_context=personalization_context,
                ),
                confidence=0.95,
            )

        if latest_qoe_score is not None and latest_qoe_score < 50.0:
            if provider == "local":
                return AdaptiveActionDecision(
                    action="CHANGE_PROVIDER",
                    reason=(
                        f"Current local provider QoE is {latest_qoe_score:.1f}; switching to the cloud provider is the safest next action."
                    ),
                    suggested_runtime_configuration=self.transform_configuration(
                        action="CHANGE_PROVIDER",
                        current_runtime_configuration=current_runtime_configuration,
                        personalization_context=personalization_context,
                    ),
                    confidence=0.9,
                )
            return AdaptiveActionDecision(
                action="CHANGE_MODEL",
                reason=(
                    f"Current QoE is {latest_qoe_score:.1f}; trying a different model is safer than repeating the same configuration."
                ),
                suggested_runtime_configuration=self.transform_configuration(
                    action="CHANGE_MODEL",
                    current_runtime_configuration=current_runtime_configuration,
                    personalization_context=personalization_context,
                ),
                confidence=0.85,
            )

        if average_recent_score is not None and average_recent_score < 60.0:
            if provider == "local":
                return AdaptiveActionDecision(
                    action="CHANGE_PROVIDER",
                    reason=(
                        f"Recent learner experience shows low average QoE ({average_recent_score:.1f}); switching away from the local provider is the safest adaptive action."
                    ),
                    suggested_runtime_configuration=self.transform_configuration(
                        action="CHANGE_PROVIDER",
                        current_runtime_configuration=current_runtime_configuration,
                        personalization_context=personalization_context,
                    ),
                    confidence=0.8,
                )
            return AdaptiveActionDecision(
                action="CHANGE_MODEL",
                reason=(
                    f"Recent learner experience shows low average QoE ({average_recent_score:.1f}); changing the model is the safest deterministic action."
                ),
                suggested_runtime_configuration=self.transform_configuration(
                    action="CHANGE_MODEL",
                    current_runtime_configuration=current_runtime_configuration,
                    personalization_context=personalization_context,
                ),
                confidence=0.75,
            )

        if latency_ms is not None and latency_ms > 2500:
            return AdaptiveActionDecision(
                action="CHANGE_MODEL",
                reason=(
                    f"Latency is high ({latency_ms} ms); switching to a lighter configuration keeps the interaction responsive."
                ),
                suggested_runtime_configuration=self.transform_configuration(
                    action="CHANGE_MODEL",
                    current_runtime_configuration=current_runtime_configuration,
                    personalization_context=personalization_context,
                ),
                confidence=0.7,
            )

        if personalization_context is not None:
            preferred_language = personalization_context.get("preferred_language")
            if preferred_language and current_runtime_configuration.get("response_language") != preferred_language:
                return AdaptiveActionDecision(
                    action="CHANGE_LANGUAGE",
                    reason=(
                        f"The learner's preferred language is {preferred_language}; aligning the response language with the learner profile is a safe adaptive action."
                    ),
                    suggested_runtime_configuration=self.transform_configuration(
                        action="CHANGE_LANGUAGE",
                        current_runtime_configuration=current_runtime_configuration,
                        personalization_context=personalization_context,
                    ),
                    confidence=0.65,
                )

            detail_level = personalization_context.get("detail_level")
            if detail_level and current_runtime_configuration.get("detail_level") != detail_level:
                return AdaptiveActionDecision(
                    action="CHANGE_RESPONSE_DETAIL",
                    reason=(
                        f"The learner prefers {detail_level} detail; adapting the response detail level aligns the answer with that preference."
                    ),
                    suggested_runtime_configuration=self.transform_configuration(
                        action="CHANGE_RESPONSE_DETAIL",
                        current_runtime_configuration=current_runtime_configuration,
                        personalization_context=personalization_context,
                    ),
                    confidence=0.65,
                )

            response_style = personalization_context.get("response_style")
            if response_style and current_runtime_configuration.get("response_style") != response_style:
                return AdaptiveActionDecision(
                    action="CHANGE_RESPONSE_STYLE",
                    reason=(
                        f"The learner prefers a {response_style} response style; applying that style is a safe adaptive action."
                    ),
                    suggested_runtime_configuration=self.transform_configuration(
                        action="CHANGE_RESPONSE_STYLE",
                        current_runtime_configuration=current_runtime_configuration,
                        personalization_context=personalization_context,
                    ),
                    confidence=0.6,
                )

        if latest_qoe_score is not None and latest_qoe_score >= 85.0:
            return None

        return None

    def transform_configuration(
        self,
        *,
        action: str,
        current_runtime_configuration: dict[str, Any] | None = None,
        personalization_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = dict(current_runtime_configuration or {})
        current_provider = str(config.get("provider", "huggingface")).lower()

        if action == "CHANGE_PROVIDER":
            config["provider"] = "local" if current_provider != "local" else "huggingface"
            if config["provider"] == "local":
                config["model_id"] = config.get("local_model_id", config.get("model_id", "local-model"))
            else:
                config["model_id"] = config.get("cloud_model_id", config.get("model_id", "local-model"))
            return config

        if action == "CHANGE_MODEL":
            if current_provider == "local":
                config["model_id"] = config.get("local_model_id", config.get("model_id", "local-model"))
            else:
                config["model_id"] = config.get("cloud_model_id", config.get("model_id", "local-model"))
            return config

        if action == "CHANGE_RETRIEVAL_K":
            top_k = int(config.get("knowledge_retrieval_top_k", 3))
            config["knowledge_retrieval_top_k"] = min(10, max(1, top_k + 1))
            return config

        if action == "CHANGE_RESPONSE_DETAIL":
            current_detail = str(config.get("detail_level", "medium")).lower()
            config["detail_level"] = "high" if current_detail != "high" else "medium"
            return config

        if action == "CHANGE_RESPONSE_STYLE":
            current_style = str(config.get("response_style", "concise")).lower()
            config["response_style"] = "step_by_step" if current_style != "step_by_step" else "concise"
            return config

        if action == "CHANGE_LANGUAGE":
            preferred_language = None
            if personalization_context is not None:
                preferred_language = personalization_context.get("preferred_language")
            config["response_language"] = (
                preferred_language
                if preferred_language is not None
                else config.get("response_language", "english")
            )
            return config

        return config

    def _select_action_from_experiences(
        self,
        *,
        learner_id: str | None,
        current_runtime_configuration: dict[str, Any],
        personalization_context: dict[str, Any] | None,
    ) -> AdaptiveActionDecision | None:
        if learner_id is None:
            return None

        experiences = self._recent_experiences(learner_id)
        if not experiences:
            return None

        candidates: list[tuple[float, str, dict[str, Any]]] = []
        for action in ALLOWED_ACTIONS:
            candidate_configuration = self.transform_configuration(
                action=action,
                current_runtime_configuration=current_runtime_configuration,
                personalization_context=personalization_context,
            )
            if candidate_configuration == current_runtime_configuration:
                continue

            utility = self._estimate_experience_utility(
                candidate_configuration=candidate_configuration,
                experiences=experiences,
            )
            if utility <= 0.0:
                continue

            candidates.append((utility, action, candidate_configuration))

        if not candidates:
            return None

        utility, action, candidate_configuration = max(candidates, key=lambda item: item[0])
        reason = (
            f"Similar historical experiences indicate that {action.lower().replace('_', ' ')} "
            f"offers the strongest observed utility ({utility:.2f})."
        )

        return AdaptiveActionDecision(
            action=action,
            reason=reason,
            suggested_runtime_configuration=candidate_configuration,
            confidence=min(0.95, max(0.55, utility / 100.0)),
        )

    def _estimate_experience_utility(
        self,
        *,
        candidate_configuration: dict[str, Any],
        experiences: list[Experience],
    ) -> float:
        weighted_reward = 0.0
        total_similarity = 0.0

        for experience in experiences:
            historical_configuration = (
                experience.configuration_snapshot.get("configuration_data")
                or experience.configuration_snapshot
                or {}
            )
            similarity = self._configuration_similarity(
                candidate_configuration,
                historical_configuration,
            )
            if similarity <= 0:
                continue

            reward_score = 0.0
            try:
                reward_score = float(experience.reward_score)
            except (TypeError, ValueError):
                continue

            weighted_reward += similarity * reward_score
            total_similarity += similarity

        if total_similarity == 0.0:
            return 0.0

        return weighted_reward / total_similarity

    def _configuration_similarity(
        self,
        candidate_configuration: dict[str, Any],
        historical_configuration: dict[str, Any],
    ) -> float:
        if not candidate_configuration or not historical_configuration:
            return 0.0

        compared_fields = [
            ("provider", 1.0),
            ("model_id", 1.0),
            ("response_language", 0.5),
            ("detail_level", 0.5),
            ("response_style", 0.5),
            ("knowledge_retrieval_top_k", 0.5),
        ]

        similarity_weight = 0.0
        total_weight = 0.0

        for field_name, weight in compared_fields:
            if field_name not in candidate_configuration and field_name not in historical_configuration:
                continue

            total_weight += weight
            if candidate_configuration.get(field_name) == historical_configuration.get(field_name):
                similarity_weight += weight

        if total_weight == 0.0:
            return 0.0

        return similarity_weight / total_weight

    def _recent_experiences(self, learner_id: str | None) -> list[Experience]:
        if learner_id is None:
            return []

        statement = (
            select(Experience)
            .where(Experience.learner_id == learner_id)
            .order_by(Experience.created_at.desc())
            .limit(20)
        )
        return list(self.repository.session.scalars(statement))

    def _recent_qoe_scores(self, learner_id: str | None) -> list[float]:
        if learner_id is None:
            return []

        experiences = self._recent_experiences(learner_id)

        scores: list[float] = []
        for experience in experiences:
            qoe_outcome = experience.qoe_outcome or {}
            qoe_score = qoe_outcome.get("score")
            if qoe_score is not None:
                try:
                    scores.append(float(qoe_score))
                except (TypeError, ValueError):
                    continue
        return scores
