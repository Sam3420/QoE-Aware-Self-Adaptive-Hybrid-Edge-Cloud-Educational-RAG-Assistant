from app.db.repositories import LearningTraceRepository
from app.domain.enums import CompetencyLevel, PreferredLanguage
from app.domain.models import (
    AssessmentResultCreate,
    AssessmentResultResponse,
    LearnerPersonalizationProfile,
    LearnerPersonalizationUpdate,
    LearningPreferences,
    PersonalizationContext,
    TopicCompetency,
)


class LearnerPersonalizationService:
    def __init__(self, repository: LearningTraceRepository) -> None:
        self.repository = repository

    def get_learner_personalization(
        self,
        learner_id: str,
    ) -> LearnerPersonalizationProfile | None:
        learner = self.repository.get_learner(learner_id)
        if learner is None:
            return None
        return LearnerPersonalizationProfile(
            learner_id=learner.id,
            preferred_language=(
                PreferredLanguage(learner.preferred_language)
                if learner.preferred_language is not None
                else None
            ),
            competency_level=(
                CompetencyLevel(learner.competency_level)
                if learner.competency_level is not None
                else None
            ),
            interests=list(learner.interests),
            learning_preferences=LearningPreferences(**learner.learning_preferences),
        )

    def update_learner_personalization(
        self,
        learner_id: str,
        data: LearnerPersonalizationUpdate,
    ) -> LearnerPersonalizationProfile:
        learner = self.repository.update_learner_personalization(learner_id, data)
        if learner is None:
            raise ValueError("Learner not found.")
        return self.get_learner_personalization(learner_id) or LearnerPersonalizationProfile(
            learner_id=learner_id,
            preferred_language=data.preferred_language,
            competency_level=data.competency_level,
            interests=data.interests,
            learning_preferences=data.learning_preferences,
        )

    def get_context_for_learner(self, learner_id: str) -> PersonalizationContext:
        learner = self.repository.get_learner(learner_id)
        if learner is None:
            raise ValueError("Learner not found.")

        topic_competencies = [
            TopicCompetency(
                topic=result.topic,
                competency_level=CompetencyLevel(result.competency_level),
                score=result.score,
            )
            for result in self.repository.list_assessment_results(learner_id)
        ]

        return PersonalizationContext(
            learner_id=learner.id,
            preferred_language=(
                PreferredLanguage(learner.preferred_language)
                if learner.preferred_language is not None
                else None
            ),
            competency_level=(
                CompetencyLevel(learner.competency_level)
                if learner.competency_level is not None
                else None
            ),
            interests=list(learner.interests),
            learning_preferences=LearningPreferences(**learner.learning_preferences),
            topic_competencies=topic_competencies,
        )

    def record_assessment_result(
        self,
        learner_id: str,
        data: AssessmentResultCreate,
    ) -> AssessmentResultResponse:
        result = self.repository.create_assessment_result(learner_id, data)
        return AssessmentResultResponse(
            id=result.id,
            learner_id=result.learner_id,
            topic=result.topic,
            competency_level=CompetencyLevel(result.competency_level),
            score=result.score,
            assessment_data=result.assessment_data,
        )
