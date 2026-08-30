from collections.abc import Generator
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.dependencies import get_db_session
from app.db.base import Base
from app.db.repositories import LearningTraceRepository
from app.domain.enums import CompetencyLevel, PreferredLanguage
from app.domain.models import LearnerProfileCreate, LearningSessionCreate
from app.main import create_app
from app.services.learning_trace_service import LearningTraceService


class FakeLLMProvider:
    def generate(self, request):
        return SimpleNamespace(
            text='A variable is a named value.',
            model_provider='fake',
            model_name=request.model_id,
        )


def test_personalization_routes_and_isolation_work():
    engine = create_engine(
        'sqlite://',
        connect_args={'check_same_thread': False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    with TestingSession() as seed_session:
        trace_service = LearningTraceService(LearningTraceRepository(seed_session))
        learner_a = trace_service.create_learner(LearnerProfileCreate(display_name='Alice'))
        learner_b = trace_service.create_learner(LearnerProfileCreate(display_name='Bob'))
        trace_service.start_session(LearningSessionCreate(learner_id=learner_a.id))
        trace_service.start_session(LearningSessionCreate(learner_id=learner_b.id))
        seed_session.commit()
        learner_a_id = learner_a.id
        learner_b_id = learner_b.id

    def override_db_session() -> Generator[Session, None, None]:
        with TestingSession() as db_session:
            yield db_session

    app = create_app()
    app.dependency_overrides[get_db_session] = override_db_session

    with TestClient(app) as client:
        put_response = client.put(
            f'/learners/{learner_a_id}/personalization',
            json={
                'preferred_language': 'hindi',
                'competency_level': 'beginner',
                'interests': ['biology'],
                'learning_preferences': {'explanation_style': 'simple', 'detail_level': 'brief'},
            },
        )
        assert put_response.status_code == 200, put_response.text
        payload = put_response.json()
        assert payload['preferred_language'] == 'hindi'
        assert payload['competency_level'] == 'beginner'
        assert payload['interests'] == ['biology']

        get_response = client.get(f'/learners/{learner_a_id}/personalization')
        assert get_response.status_code == 200, get_response.text
        assert get_response.json()['preferred_language'] == 'hindi'

        other_response = client.get(f'/learners/{learner_b_id}/personalization')
        assert other_response.status_code == 200, other_response.text
        assert other_response.json()['preferred_language'] is None

        assessment_response = client.post(
            f'/learners/{learner_a_id}/assessments',
            json={
                'topic': 'biology',
                'competency_level': 'intermediate',
                'score': 80,
                'assessment_data': {'source': 'pretest'},
            },
        )
        assert assessment_response.status_code == 200, assessment_response.text
        assert assessment_response.json()['topic'] == 'biology'

    with TestingSession() as assert_session:
        repo = LearningTraceRepository(assert_session)
        profile_a = repo.get_learner(learner_a_id)
        profile_b = repo.get_learner(learner_b_id)
        assert profile_a is not None
        assert profile_b is not None
        assert profile_a.preferred_language == PreferredLanguage.HINDI.value
        assert profile_b.preferred_language is None
        assessment_rows = repo.list_assessment_results(learner_a.id)
        assert len(assessment_rows) == 1
        assert assessment_rows[0].topic == 'biology'
        assert assessment_rows[0].competency_level == CompetencyLevel.INTERMEDIATE.value
