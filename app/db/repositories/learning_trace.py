from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, joinedload

from app.db.models import (
    EducationalResource,
    AssessmentResult,
    Interaction,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeIndex,
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
    LearnerPersonalizationUpdate,
    AssessmentResultCreate,
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

    def update_learner_personalization(
        self,
        learner_id: str,
        data: LearnerPersonalizationUpdate,
    ) -> LearnerProfile | None:
        learner = self.get_learner(learner_id)
        if learner is None:
            return None
        learner.preferred_language = (
            data.preferred_language.value if data.preferred_language else None
        )
        learner.competency_level = (
            data.competency_level.value if data.competency_level else None
        )
        learner.interests = data.interests
        learner.learning_preferences = data.learning_preferences.model_dump(
            exclude_none=True
        )
        self.session.flush()
        return learner

    def create_assessment_result(
        self,
        learner_id: str,
        data: AssessmentResultCreate,
    ) -> AssessmentResult:
        result = AssessmentResult(
            learner_id=learner_id,
            topic=data.topic,
            competency_level=data.competency_level.value,
            score=data.score,
            assessment_data=data.assessment_data,
        )
        self.session.add(result)
        self.session.flush()
        return result

    def list_assessment_results(self, learner_id: str) -> list[AssessmentResult]:
        statement = (
            select(AssessmentResult)
            .where(AssessmentResult.learner_id == learner_id)
            .order_by(AssessmentResult.created_at.desc())
        )
        return list(self.session.scalars(statement))

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

    def get_runtime_configuration(
        self,
        runtime_configuration_id: str,
    ) -> RuntimeConfiguration | None:
        return self.session.get(RuntimeConfiguration, runtime_configuration_id)

    def get_latest_runtime_configuration_by_name(
        self,
        name: str,
    ) -> RuntimeConfiguration | None:
        statement = (
            select(RuntimeConfiguration)
            .where(RuntimeConfiguration.name == name)
            .order_by(RuntimeConfiguration.version.desc())
            .limit(1)
        )
        return self.session.scalars(statement).first()

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

    def get_or_create_runtime_configuration_snapshot(
        self,
        *,
        name: str,
        description: str | None,
        configuration_data: dict,
    ) -> RuntimeConfiguration:
        latest = self.get_latest_runtime_configuration_by_name(name)
        if (
            latest is not None
            and latest.description == description
            and latest.configuration_data == configuration_data
        ):
            return latest

        return self.create_runtime_configuration_version(
            name=name,
            description=description,
            configuration_data=configuration_data,
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

    def get_educational_resource_by_source_and_external_id(
        self,
        *,
        source: str,
        external_resource_id: str,
    ) -> EducationalResource | None:
        statement = select(EducationalResource).where(
            EducationalResource.source == source,
            EducationalResource.external_resource_id == external_resource_id,
        )
        return self.session.scalars(statement).first()

    def get_educational_resource(self, resource_id: str) -> EducationalResource | None:
        return self.session.get(EducationalResource, resource_id)

    def create_knowledge_document(
        self,
        *,
        resource_id: str,
        title: str,
        source: str,
        transcript_text: str,
        metadata: dict | None = None,
    ) -> KnowledgeDocument:
        document = KnowledgeDocument(
            resource_id=resource_id,
            title=title,
            source=source,
            transcript_text=transcript_text,
            metadata=metadata or {},
            status="ready",
        )
        self.session.add(document)
        self.session.flush()
        return document

    def get_knowledge_document_for_resource(self, resource_id: str) -> KnowledgeDocument | None:
        statement = select(KnowledgeDocument).where(KnowledgeDocument.resource_id == resource_id)
        return self.session.scalars(statement).first()

    def create_knowledge_chunk(
        self,
        *,
        document_id: str | None,
        chunk_index: int,
        content: str,
        metadata: dict | None = None,
    ) -> KnowledgeChunk:
        chunk = KnowledgeChunk(
            document_id=document_id or "",
            chunk_index=chunk_index,
            content=content,
            metadata=metadata or {},
            status="ready",
        )
        self.session.add(chunk)
        self.session.flush()
        return chunk

    def attach_document_chunks(self, document_id: str, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            chunk = self.session.get(KnowledgeChunk, chunk_id)
            if chunk is not None:
                chunk.document_id = document_id
        self.session.flush()

    def get_knowledge_chunk(self, chunk_id: str) -> KnowledgeChunk | None:
        return self.session.get(KnowledgeChunk, chunk_id)

    def create_knowledge_index(
        self,
        *,
        document_id: str,
        index_name: str,
        index_path: str,
        embedding_model: str,
        chunk_count: int,
        metadata: dict | None = None,
    ) -> KnowledgeIndex:
        knowledge_index = KnowledgeIndex(
            document_id=document_id,
            index_name=index_name,
            index_path=index_path,
            embedding_model=embedding_model,
            chunk_count=chunk_count,
            status="ready",
            metadata=metadata or {},
        )
        self.session.add(knowledge_index)
        self.session.flush()
        return knowledge_index

    def update_knowledge_document_status(self, document_id: str, status: str) -> KnowledgeDocument | None:
        document = self.session.get(KnowledgeDocument, document_id)
        if document is None:
            return None
        document.status = status
        self.session.flush()
        return document

    def _next_runtime_configuration_version(self, name: str) -> int:
        current_version = self.session.scalar(
            select(func.max(RuntimeConfiguration.version)).where(
                RuntimeConfiguration.name == name,
            )
        )
        return 1 if current_version is None else current_version + 1
