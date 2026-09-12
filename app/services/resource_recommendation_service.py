from __future__ import annotations

from app.db.repositories import LearningTraceRepository
from app.domain.enums import CompetencyLevel
from app.domain.models import (
    PersonalizationContext,
    ResourceRecommendationItem,
    ResourceRecommendationRequest,
    ResourceRecommendationResponse,
    ResourceSelectionResponse,
)
from app.services.errors import (
    LearnerNotFoundError,
    LearningSessionNotFoundError,
    NoRecommendationsFoundError,
    ResourceNotFoundError,
)
from app.youtube.provider import YouTubeProvider


class ResourceRecommendationService:
    def __init__(self, *, repository: LearningTraceRepository, youtube_provider: YouTubeProvider) -> None:
        # Keep the external provider replaceable so ranking can be tested independently.
        self.repository = repository
        self.youtube_provider = youtube_provider
    '''
    This function:

    Loads the learning session.
    Finds the learner.
    Builds personalization context.
    Determines the preferred language.
    Searches YouTube.
    Calls _rank_candidates.
    Returns ranked resources.
    '''
    def get_recommendations_for_session(
        self,
        *,
        session_id: str,
        request: ResourceRecommendationRequest,
    ) -> ResourceRecommendationResponse:
        # The session identifies the learner whose preferences control ranking. that is the current learner for the session.
        learning_session = self.repository.get_learning_session(session_id)
        if learning_session is None:
            raise LearningSessionNotFoundError("Learning session was not found.")
        learner = self.repository.get_learner(learning_session.learner_id)
        if learner is None:#validate sess exist
            raise LearnerNotFoundError("Learner not found.")
        # Fetch candidates first; the YouTube API supplies metadata, not transcripts.
        #learner-specific information.The context might contain:preferred language,interests,competency level
        personalization_context = self._build_personalization_context(learning_session.learner_id)
        language = self._normalize_language(
            request.preferred_language.value
            if request.preferred_language is not None
            else personalization_context.preferred_language.value
            if personalization_context.preferred_language is not None
            else "en"
        )
        #fetching data
        candidates = self.youtube_provider.search_educational_videos(
            query=request.topic,
            language=language,
            max_results=request.max_results,
        )
        # Ranking is local and deterministic, so provider ordering does not decide the result.
        ranked_items = self._rank_candidates(
            candidates=candidates,
            topic=request.topic,
            personalization_context=personalization_context,
        )
        if not ranked_items:
            raise NoRecommendationsFoundError("No recommendations were found for the requested topic.")

        return ResourceRecommendationResponse(
            session_id=session_id,
            learner_id=learning_session.learner_id,
            topic=request.topic,
            recommendations=ranked_items,
        )

    def select_resource_for_session(
        self,
        *,
        session_id: str,
        resource_id: str,
    ) -> ResourceSelectionResponse:
        # Selection validates that the chosen resource belongs to the supported provider.
        learning_session = self.repository.get_learning_session(session_id)
        if learning_session is None:
            raise LearningSessionNotFoundError("Learning session was not found.")

        resource = self.repository.get_educational_resource(resource_id)
        if resource is None:
            raise ResourceNotFoundError("Educational resource was not found.")
        #check provider
        if resource.source != "youtube":
            raise ResourceNotFoundError("Educational resource is not from the supported provider.")
        #returning data abt selected video
        return ResourceSelectionResponse(
            session_id=session_id,
            learner_id=learning_session.learner_id,
            resource_id=resource.id,
            title=resource.title,
            url=resource.url or "",
        )
    #private helper function
    def _build_personalization_context(self, learner_id: str) -> PersonalizationContext:
        # Keep profile-to-context conversion in the personalization service.
        learner = self.repository.get_learner(learner_id)#find learner
        if learner is None:
            raise LearnerNotFoundError("Learner not found.")

        from app.services.personalization_service import LearnerPersonalizationService

        service = LearnerPersonalizationService(self.repository)
        return service.get_context_for_learner(learner_id)

    '''
    one of the most important functions in the service, it ranks the candidates based on multiple factors and returns the top 5 recommendations.

    The ranking score is based on:

    preferred language match
    topic match in title or description
    learner interest match
    competency level
    channel metadata
    title availability
    '''
    def _rank_candidates(
        self,
        *,
        candidates: list[dict],
        topic: str,
        personalization_context: PersonalizationContext,
    ) -> list[ResourceRecommendationItem]:
        # Normalize comparison values so ranking is case-insensitive.
        topic_lower = topic.lower()
        interests = {i.lower() for i in personalization_context.interests}
        preferred_language = self._normalize_language(
            personalization_context.preferred_language.value
            if personalization_context.preferred_language is not None
            else "en"
        )
        competency = personalization_context.competency_level

        ranked: list[ResourceRecommendationItem] = []
        for candidate in candidates:
            title = (candidate.get("title") or "").lower()
            channel = (candidate.get("channel_title") or "").lower()
            language = self._normalize_language(candidate.get("language") or preferred_language)
            description = (candidate.get("description") or "").lower()

            # Each matching signal adds an explainable contribution to the score.
            score = 0.0
            reasons: list[str] = []

            if language == preferred_language:
                score += 40.0
                reasons.append("preferred language match")#language score
            if topic_lower in title or topic_lower in description:
                score += 35.0
                reasons.append("topic match")#topic score
            if any(interest in title or interest in description for interest in interests):
                score += 15.0
                reasons.append("interest match")
            if competency is not None:
                if competency == CompetencyLevel.BEGINNER:
                    score += 8.0
                    reasons.append("beginner-friendly content")
                elif competency == CompetencyLevel.INTERMEDIATE:
                    score += 10.0
                    reasons.append("matched intermediate level")
                else:
                    score += 12.0
                    reasons.append("advanced-level fit")
            if candidate.get("channel_title"):
                score += 3.0
                reasons.append("channel metadata available")
            if candidate.get("title"):
                score += 1.0

            if score <= 0:
                continue
            #construct resource id
            resource_id = f"youtube:{candidate['external_id']}"
            ranked.append(
                ResourceRecommendationItem(
                    resource_id=resource_id,
                    provider=candidate.get("provider") or "youtube",
                    external_id=candidate.get("external_id") or "",
                    title=candidate.get("title") or "Untitled video",
                    url=candidate.get("url") or "",
                    channel_title=candidate.get("channel_title"),
                    ranking_score=round(score, 2),
                    relevance_reason=", ".join(reasons) if reasons else "general educational match",
                )
            )

        ranked.sort(key=lambda item: (-item.ranking_score, item.title.lower()))
        return ranked[: max(1, min(len(ranked), 5))]

    @staticmethod
    def _normalize_language(language: str | None) -> str:
        if language is None:
            return "en"
        normalized = language.strip().lower()
        mapping = {
            "english": "en",
            "hindi": "hi"
        }
        # Unknown language values fall back to English unless they are valid two-letter codes.
        return mapping.get(normalized, normalized if len(normalized) == 2 else "en")
