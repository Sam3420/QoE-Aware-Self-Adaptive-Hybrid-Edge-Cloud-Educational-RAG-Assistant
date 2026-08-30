from dataclasses import dataclass
from time import perf_counter

from app.core.config import Settings
from app.core.logging import get_logger
from app.db.models import RuntimeConfiguration
from app.db.repositories import LearningTraceRepository
from app.domain.enums import InteractionStatus
from app.domain.models import InteractionCreate
from app.llm.provider import LLMProvider
from app.llm.types import LLMGenerateRequest, LLMProviderError
from app.services.crag_service import RetrievalQualityService
from app.services.errors import (
    AssistantProviderUnavailableError,
    LearningSessionNotFoundError,
    RuntimeConfigurationNotFoundError,
)
from app.services.personalization_service import LearnerPersonalizationService


class KnowledgeRetrievalServiceProtocol:
    def retrieve_context(self, *, resource_id: str, query: str, top_k: int | None = None):
        ...

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful educational assistant. Answer clearly and concisely, "
    "using learner-friendly language."
)


@dataclass(frozen=True)
class AssistantResponse:
    interaction_id: str
    session_id: str
    answer: str
    response_latency_ms: int
    model_provider: str
    model_name: str
    runtime_configuration_id: str


class AssistantService:
    def __init__(
        self,
        *,
        repository: LearningTraceRepository,
        llm_provider: LLMProvider,
        settings: Settings,
        personalization_service: LearnerPersonalizationService | None = None,
        knowledge_retrieval_service: KnowledgeRetrievalServiceProtocol | None = None,
        retrieval_quality_service: RetrievalQualityService | None = None,
    ) -> None:
        self.repository = repository
        self.llm_provider = llm_provider
        self.settings = settings
        self.personalization_service = personalization_service or LearnerPersonalizationService(
            repository
        )
        self.knowledge_retrieval_service = knowledge_retrieval_service
        self.retrieval_quality_service = retrieval_quality_service or RetrievalQualityService(
            min_score=settings.crag_quality_min_score,
        )

    def answer_text_question(
        self,
        *,
        session_id: str,
        question: str,
        runtime_configuration_id: str | None = None,
        resource_id: str | None = None,
    ) -> AssistantResponse:
        learning_session = self.repository.get_learning_session(session_id)
        if learning_session is None:
            raise LearningSessionNotFoundError("Learning session was not found.")

        runtime_configuration = self._resolve_runtime_configuration(runtime_configuration_id)
        personalization_context = self._build_personalization_context(learning_session.learner_id)
        retrieval_context = self._retrieve_context_for_question(question=question, resource_id=resource_id)
        llm_request = self._build_llm_request(
            question,
            runtime_configuration,
            personalization_context,
            retrieval_context=retrieval_context,
        )

        started_at = perf_counter()
        try:
            llm_result = self.llm_provider.generate(llm_request)
        except LLMProviderError as exc:
            latency_ms = self._elapsed_ms(started_at)
            interaction = self.repository.create_interaction(
                InteractionCreate(
                    session_id=session_id,
                    runtime_configuration_id=runtime_configuration.id,
                    user_input=question,
                    response_latency_ms=latency_ms,
                    model_provider="huggingface",
                    model_name=llm_request.model_id,
                    status=InteractionStatus.FAILED,
                    error_message="LLM provider failed to generate a response.",
                )
            )
            logger.warning(
                "assistant_provider_failed session_id=%s interaction_id=%s runtime_configuration_id=%s",
                session_id,
                interaction.id,
                runtime_configuration.id,
            )
            raise AssistantProviderUnavailableError(
                "Assistant provider is currently unavailable.",
                interaction_id=interaction.id,
            ) from exc

        latency_ms = self._elapsed_ms(started_at)
        interaction = self.repository.create_interaction(
            InteractionCreate(
                session_id=session_id,
                runtime_configuration_id=runtime_configuration.id,
                user_input=question,
                assistant_output=llm_result.text,
                response_latency_ms=latency_ms,
                model_provider=llm_result.model_provider,
                model_name=llm_result.model_name,
                status=InteractionStatus.COMPLETED,
            )
        )
        logger.info(
            "assistant_interaction_completed session_id=%s interaction_id=%s runtime_configuration_id=%s",
            session_id,
            interaction.id,
            runtime_configuration.id,
        )

        return AssistantResponse(
            interaction_id=interaction.id,
            session_id=session_id,
            answer=llm_result.text,
            response_latency_ms=latency_ms,
            model_provider=llm_result.model_provider,
            model_name=llm_result.model_name,
            runtime_configuration_id=runtime_configuration.id,
        )

    def _resolve_runtime_configuration(
        self,
        runtime_configuration_id: str | None,
    ) -> RuntimeConfiguration:
        if runtime_configuration_id is not None:
            runtime_configuration = self.repository.get_runtime_configuration(
                runtime_configuration_id,
            )
            if runtime_configuration is None:
                raise RuntimeConfigurationNotFoundError(
                    "Runtime configuration was not found."
                )
            return runtime_configuration

        return self.repository.get_or_create_runtime_configuration_snapshot(
            name=self.settings.default_runtime_configuration_name,
            description="Default Hugging Face text assistant configuration.",
            configuration_data=self._default_runtime_configuration_data(),
        )

    def _default_runtime_configuration_data(self) -> dict:
        return {
            "provider": "huggingface",
            "model_id": self.settings.llm_model_id,
            "max_tokens": self.settings.llm_max_tokens,
            "temperature": self.settings.llm_temperature,
        }

    def _build_personalization_context(self, learner_id: str):
        return self.personalization_service.get_context_for_learner(learner_id)

    def _build_llm_request(
        self,
        question: str,
        runtime_configuration: RuntimeConfiguration,
        personalization_context=None,
        retrieval_context: str | None = None,
    ) -> LLMGenerateRequest:
        configuration_data = runtime_configuration.configuration_data
        system_prompt = self._build_system_prompt(personalization_context, retrieval_context)
        return LLMGenerateRequest(
            prompt=question,
            system_prompt=system_prompt,
            model_id=configuration_data.get("model_id", self.settings.llm_model_id),
            max_tokens=configuration_data.get("max_tokens", self.settings.llm_max_tokens),
            temperature=configuration_data.get("temperature", self.settings.llm_temperature),
        )

    def _retrieve_context_for_question(self, *, question: str, resource_id: str | None) -> str | None:
        if resource_id is None or self.knowledge_retrieval_service is None:
            return None
        hits = self.knowledge_retrieval_service.retrieve_context(
            resource_id=resource_id,
            query=question,
            top_k=self.settings.knowledge_retrieval_top_k,
        )
        if not hits:
            return None

        decision = self.retrieval_quality_service.evaluate(question=question, hits=hits)
        if not decision.is_adequate:
            return (
                "No sufficiently relevant retrieved context was found for this question. "
                "Answer using your general knowledge only if it is clearly supported by the learner's context, "
                "and say when the retrieved material is insufficient."
            )

        context = "\n".join(f"- {hit['content']}" for hit in hits[:3])
        return (
            "Use the following retrieved educational context to answer the learner, "
            f"and do not invent unsupported facts. If the context does not contain enough information, "
            f"say so clearly.\n{context}"
        )

    def _build_system_prompt(self, personalization_context=None, retrieval_context: str | None = None) -> str:
        base_prompt = SYSTEM_PROMPT
        if personalization_context is None:
            return base_prompt

        language_instruction = ""
        if personalization_context.preferred_language is not None:
            language_name = personalization_context.preferred_language.value.title()
            language_instruction = f" Respond in {language_name} language."

        competency_instruction = ""
        if personalization_context.competency_level is not None:
            competency = personalization_context.competency_level.value
            if competency == "beginner":
                competency_instruction = " Explain in a simple, beginner-friendly way with short steps and clear examples."
            elif competency == "intermediate":
                competency_instruction = " Explain at an intermediate level with balanced detail and practical examples."
            else:
                competency_instruction = " Explain with precise, technical detail suitable for an advanced learner."

        interest_instruction = ""
        if personalization_context.interests:
            interest_instruction = (
                f" Align examples and explanations to the learner's interests: {', '.join(personalization_context.interests)}."
            )

        if retrieval_context:
            return f"{base_prompt}{language_instruction}{competency_instruction}{interest_instruction} {retrieval_context}".strip()
        return f"{base_prompt}{language_instruction}{competency_instruction}{interest_instruction}".strip()

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return max(0, int((perf_counter() - started_at) * 1000))
