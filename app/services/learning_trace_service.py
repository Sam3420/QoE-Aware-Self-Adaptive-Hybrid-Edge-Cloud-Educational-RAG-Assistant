from app.core.logging import get_logger
from app.db.models import (
    EducationalResource,
    Interaction,
    LearnerProfile,
    LearningSession,
    RuntimeConfiguration,
)
from app.db.repositories import LearningTraceRepository
from app.domain.models import (
    EducationalResourceCreate,
    InteractionCreate,
    LearnerProfileCreate,
    LearningSessionCreate,
)

logger = get_logger(__name__)


class LearningTraceService:
    def __init__(self, repository: LearningTraceRepository) -> None:
        self.repository = repository

    def _coerce_model(self, data, model_type):
        if isinstance(data, model_type):
            return data
        if isinstance(data, dict):
            return model_type(**data)
        return data

    def create_learner(self, data: LearnerProfileCreate) -> LearnerProfile:
        coerced = self._coerce_model(data, LearnerProfileCreate)
        learner = self.repository.create_learner(coerced)
        logger.info("learner_created learner_id=%s", learner.id)
        return learner

    def start_session(self, data: LearningSessionCreate) -> LearningSession:
        coerced = self._coerce_model(data, LearningSessionCreate)
        learning_session = self.repository.create_learning_session(coerced)
        logger.info(
            "session_started learner_id=%s session_id=%s",
            learning_session.learner_id,
            learning_session.id,
        )
        return learning_session

    def create_resource(self, data: EducationalResourceCreate) -> EducationalResource:
        coerced = self._coerce_model(data, EducationalResourceCreate)
        resource = self.repository.create_educational_resource(coerced)
        logger.info("resource_created resource_id=%s source=%s", resource.id, resource.source)
        return resource

    def create_runtime_configuration_snapshot(
        self,
        *,
        name: str,
        description: str | None,
        configuration_data: dict,
    ) -> RuntimeConfiguration:
        runtime_configuration = self.repository.create_runtime_configuration_version(
            name=name,
            description=description,
            configuration_data=configuration_data,
        )
        logger.info(
            "runtime_configuration_snapshot_created runtime_configuration_id=%s name=%s version=%s",
            runtime_configuration.id,
            runtime_configuration.name,
            runtime_configuration.version,
        )
        return runtime_configuration

    def record_interaction(self, data: InteractionCreate) -> Interaction:
        interaction = self.repository.create_interaction(data)
        logger.info(
            "interaction_recorded session_id=%s interaction_id=%s runtime_configuration_id=%s",
            interaction.session_id,
            interaction.id,
            interaction.runtime_configuration_id,
        )
        return interaction
