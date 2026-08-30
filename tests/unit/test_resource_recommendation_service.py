from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models import LearningSession
from app.db.repositories import LearningTraceRepository
from app.domain.enums import CompetencyLevel, PreferredLanguage
from app.domain.models import (
    LearnerPersonalizationUpdate,
    LearnerProfileCreate,
    LearningSessionCreate,
    ResourceRecommendationRequest,
)
from app.services.learning_trace_service import LearningTraceService
from app.services.personalization_service import LearnerPersonalizationService
from app.services.resource_recommendation_service import ResourceRecommendationService


class FakeYouTubeProvider:
    def __init__(self, results):
        self.results = results

    def search_educational_videos(self, *, query: str, language: str | None, max_results: int):
        return [dict(item) for item in self.results[:max_results]]


def test_recommendation_service_ranks_by_language_and_interest():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    with Session() as session:
        trace = LearningTraceService(LearningTraceRepository(session))
        learner = trace.create_learner(LearnerProfileCreate(display_name="Sam"))
        learning_session = trace.start_session(LearningSessionCreate(learner_id=learner.id))

        personalization = LearnerPersonalizationService(LearningTraceRepository(session))
        personalization.update_learner_personalization(
            learner.id,
            LearnerPersonalizationUpdate(
                preferred_language=PreferredLanguage.ENGLISH,
                competency_level=CompetencyLevel.BEGINNER,
                interests=["biology"],
            ),
        )
        session.commit()

        provider = FakeYouTubeProvider(
            [
                {
                    "provider": "youtube",
                    "external_id": "z1",
                    "title": "Biology basics for beginners",
                    "url": "https://example.com/z1",
                    "channel_title": "Study Lab",
                    "description": "Intro biology explained simply",
                    "language": "en",
                },
                {
                    "provider": "youtube",
                    "external_id": "z2",
                    "title": "Advanced robotics lecture",
                    "url": "https://example.com/z2",
                    "channel_title": "Tech Academy",
                    "description": "Robotics overview",
                    "language": "en",
                },
            ]
        )
        recommendation_service = ResourceRecommendationService(
            repository=LearningTraceRepository(session),
            youtube_provider=provider,
        )

        recommendation_response = recommendation_service.get_recommendations_for_session(
            session_id=learning_session.id,
            request=ResourceRecommendationRequest(topic="biology", max_results=5),
        )

        assert recommendation_response.recommendations[0].external_id == "z1"
        assert recommendation_response.recommendations[0].ranking_score >= 80


def test_recommendation_service_raises_when_no_results_available():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    with Session() as session:
        trace = LearningTraceService(LearningTraceRepository(session))
        learner = trace.create_learner(LearnerProfileCreate(display_name="Sam"))
        learning_session = trace.start_session(LearningSessionCreate(learner_id=learner.id))
        session.commit()

        provider = FakeYouTubeProvider([])
        recommendation_service = ResourceRecommendationService(
            repository=LearningTraceRepository(session),
            youtube_provider=provider,
        )

        try:
            recommendation_service.get_recommendations_for_session(
                session_id=learning_session.id,
                request=ResourceRecommendationRequest(topic="quantum physics", max_results=5),
            )
            raise AssertionError("Expected NoRecommendationsFoundError to be raised.")
        except Exception as exc:
            assert exc.__class__.__name__ == "NoRecommendationsFoundError"
