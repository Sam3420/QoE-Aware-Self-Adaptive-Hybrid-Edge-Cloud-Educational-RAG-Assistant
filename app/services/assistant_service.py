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
from app.services.errors import (
    AssistantProviderUnavailableError,
    LearningSessionNotFoundError,
    RuntimeConfigurationNotFoundError,
)

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
    ) -> None:
        self.repository = repository
        self.llm_provider = llm_provider
        self.settings = settings

    def answer_text_question(
        self,
        *,
        session_id: str,
        question: str,
        runtime_configuration_id: str | None = None,
    ) -> AssistantResponse:
        learning_session = self.repository.get_learning_session(session_id)
        if learning_session is None:
            raise LearningSessionNotFoundError("Learning session was not found.")

        runtime_configuration = self._resolve_runtime_configuration(runtime_configuration_id)
        llm_request = self._build_llm_request(question, runtime_configuration)

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

    def _build_llm_request(
        self,
        question: str,
        runtime_configuration: RuntimeConfiguration,
    ) -> LLMGenerateRequest:
        configuration_data = runtime_configuration.configuration_data
        return LLMGenerateRequest(
            prompt=question,
            system_prompt=SYSTEM_PROMPT,
            model_id=configuration_data.get("model_id", self.settings.llm_model_id),
            max_tokens=configuration_data.get("max_tokens", self.settings.llm_max_tokens),
            temperature=configuration_data.get("temperature", self.settings.llm_temperature),
        )

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return max(0, int((perf_counter() - started_at) * 1000))
