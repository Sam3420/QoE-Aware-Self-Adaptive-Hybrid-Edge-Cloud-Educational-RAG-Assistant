from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, joinedload

from app.db.models import (
    EducationalResource,
    Interaction,
    LearnerProfile,
    LearningSession,
    RuntimeConfiguration,
)
from app.domain.models import (
    EducationalResourceCreate,
    InteractionCreate,
    LearnerProfileCreate,
    LearningSessionCreate,
    RuntimeConfigurationCreate,
)


class LearningTraceRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_learner(self, data: LearnerProfileCreate) -> LearnerProfile:
        learner = LearnerProfile(
            display_name=data.display_name,
            profile_data=data.profile_data,
        )
        self.session.add(learner)
        self.session.flush()
        return learner

    def get_learner(self, learner_id: str) -> LearnerProfile | None:
        return self.session.get(LearnerProfile, learner_id)

    def create_learning_session(self, data: LearningSessionCreate) -> LearningSession:
        learning_session = LearningSession(
            learner_id=data.learner_id,
            status=data.status.value,
        )
        self.session.add(learning_session)
        self.session.flush()
        return learning_session

    def get_learning_session(self, session_id: str) -> LearningSession | None:
        statement: Select[tuple[LearningSession]] = (
            select(LearningSession)
            .where(LearningSession.id == session_id)
            .options(joinedload(LearningSession.learner))
        )
        return self.session.scalars(statement).first()

    def create_educational_resource(
        self,
        data: EducationalResourceCreate,
    ) -> EducationalResource:
        resource = EducationalResource(
            source=data.source,
            external_resource_id=data.external_resource_id,
            title=data.title,
            url=str(data.url) if data.url else None,
            topic=data.topic,
            resource_metadata=data.metadata,
        )
        self.session.add(resource)
        self.session.flush()
        return resource

    def create_runtime_configuration(
        self,
        data: RuntimeConfigurationCreate,
    ) -> RuntimeConfiguration:
        runtime_configuration = RuntimeConfiguration(
            name=data.name,
            version=data.version,
            description=data.description,
            configuration_data=data.configuration_data,
        )
        self.session.add(runtime_configuration)
        self.session.flush()
        return runtime_configuration

    def create_runtime_configuration_version(
        self,
        *,
        name: str,
        description: str | None,
        configuration_data: dict,
    ) -> RuntimeConfiguration:
        next_version = self._next_runtime_configuration_version(name)
        return self.create_runtime_configuration(
            RuntimeConfigurationCreate(
                name=name,
                version=next_version,
                description=description,
                configuration_data=configuration_data,
            )
        )

    def create_interaction(self, data: InteractionCreate) -> Interaction:
        interaction_data = {
            "session_id": data.session_id,
            "runtime_configuration_id": data.runtime_configuration_id,
            "educational_resource_id": data.educational_resource_id,
            "user_input": data.user_input,
            "assistant_output": data.assistant_output,
            "response_latency_ms": data.response_latency_ms,
            "model_provider": data.model_provider,
            "model_name": data.model_name,
            "status": data.status.value,
            "error_message": data.error_message,
        }
        if data.occurred_at is not None:
            interaction_data["occurred_at"] = data.occurred_at

        interaction = Interaction(**interaction_data)
        self.session.add(interaction)
        self.session.flush()
        return interaction

    def get_interaction_with_trace(self, interaction_id: str) -> Interaction | None:
        statement = (
            select(Interaction)
            .where(Interaction.id == interaction_id)
            .options(
                joinedload(Interaction.session).joinedload(LearningSession.learner),
                joinedload(Interaction.runtime_configuration),
            )
        )
        return self.session.scalars(statement).first()

    def _next_runtime_configuration_version(self, name: str) -> int:
        current_version = self.session.scalar(
            select(func.max(RuntimeConfiguration.version)).where(
                RuntimeConfiguration.name == name,
            )
        )
        return 1 if current_version is None else current_version + 1
