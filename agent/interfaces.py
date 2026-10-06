"""Gateway from the agent to Person B's code in app/ (repository, pre/post-test, experience storage).

Nodes call only this module, never app/ directly, and never SQLite. Each call opens one DB
session, passes it to B's function (so B's code does not open its own) and commits.
Functions B has not written yet are clearly marked STUB at the bottom.
"""
import json
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings as get_app_settings
from app.db.base import Base
from app.db.repositories import AdaptationRepository, LearningTraceRepository
from app.domain.models import LearnerProfileCreate, LearningSessionCreate
from app.llm.types import LLMGenerateRequest, LLMGenerateResult, LLMProviderError
from app.services import experience as b_experience
from app.services import posttest as b_posttest
from app.services import pretest as b_pretest
from services.crag import CragRetriever
from services.llm_clients import GenerationRequest, LLMError
from services.llm_router import LLMRouter

logger = logging.getLogger(__name__)


class LearnerNotFound(LookupError):
    """Raised when a session is started for a learner that does not exist."""


# ---- Adapters B's MCQ generation needs (retriever + LLMProvider protocols) ---------------------

class CragKnowledgeRetriever:
    """Adapts the FAISS store to B's KnowledgeRetriever protocol (retrieve_context → [{content, ...}])."""

    def __init__(self, crag: CragRetriever) -> None:
        self.crag = crag

    def retrieve_context(self, *, resource_id: str, query: str, top_k: int | None = None) -> list[dict]:
        """Return plain-dict hits with chunk text under 'content'."""
        hits = self.crag.store.search(resource_id, query, top_k or 5)
        return [{"chunk_id": h.chunk_id, "content": h.text, "score": h.score} for h in hits]


def mcq_json_schema(n_items: int) -> dict[str, Any]:
    """JSON schema for B's MCQ reply: an array of n {question, options[4], answer_index 0-3, concept}."""
    item = {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4},
            "answer_index": {"type": "integer", "minimum": 0, "maximum": 3},
            "concept": {"type": "string"},
        },
        "required": ["question", "options", "answer_index", "concept"],
    }
    return {"type": "array", "items": item, "minItems": n_items, "maxItems": n_items}


class RouterLLMProvider:
    """Adapts the LLM router to B's LLMProvider protocol (used for MCQ generation): cloud if available, else local.

    Small local models write malformed JSON, so requests carry B's MCQ schema; Ollama enforces it.
    """

    def __init__(self, router: LLMRouter) -> None:
        self.router = router

    def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        """Generate with the best available real model; raise LLMProviderError if none works."""
        req = GenerationRequest(system=request.system_prompt or "", prompt=request.prompt,
                                max_tokens=request.max_tokens, temperature=request.temperature,
                                json_schema=mcq_json_schema(get_app_settings().mcq_n_items))
        errors = []
        for provider in ("cloud", "local"):
            if not self.router.is_available(provider):
                continue
            try:
                result = self.router.generate(provider, req)
                return LLMGenerateResult(text=result.text, model_provider=result.provider, model_name=result.model)
            except LLMError as exc:
                errors.append(str(exc))
        raise LLMProviderError("No LLM could generate MCQs: " + ("; ".join(errors) or "none available"))


class FakeMCQProvider:
    """Offline LLMProvider returning 5 valid MCQs (correct letters A, B, C, D, A)."""

    def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        """Return a fixed JSON array of MCQ items."""
        items = [
            {"question": f"Practice question {i + 1}?", "options": [f"option {i}-{c}" for c in "abcd"],
             "answer_index": i % 4, "concept": f"concept {i + 1}"}
            for i in range(5)
        ]
        return LLMGenerateResult(text=json.dumps(items), model_provider="fake", model_name="fake-mcq")


# ---- Gateway -------------------------------------------------------------------------------------

class AppGateway:
    """All agent → Person B calls, with an injectable DB session factory."""

    def __init__(self, session_factory: Callable[[], Session], *, retriever: Any, mcq_llm: Any) -> None:
        self.session_factory = session_factory
        self.retriever = retriever
        self.mcq_llm = mcq_llm
        self._ready = False

    @contextmanager
    def _db(self) -> Iterator[Session]:
        """Open a session, commit on success, roll back on error."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def ensure_ready(self) -> None:
        """Create B's tables if missing and seed the predefined policies (idempotent)."""
        if self._ready:
            return
        with self._db() as db:
            Base.metadata.create_all(bind=db.get_bind())
            AdaptationRepository(db).seed_policies()
        self._ready = True

    # learners and sessions (B's LearningTraceRepository)
    def create_learner(self, display_name: str) -> str:
        """Create a learner and return its id (scripts and tests; the UI owns real sign-up)."""
        with self._db() as db:
            return LearningTraceRepository(db).create_learner(LearnerProfileCreate(display_name=display_name)).id

    def create_learning_session(self, student_id: str) -> str:
        """Create the LearningSession row B's writes require and return its id."""
        with self._db() as db:
            trace = LearningTraceRepository(db)
            if trace.get_learner(student_id) is None:
                raise LearnerNotFound(f"Learner {student_id!r} does not exist.")
            return trace.create_learning_session(LearningSessionCreate(learner_id=student_id)).id

    def get_learner_profile(self, student_id: str) -> dict[str, Any]:
        """Return the learner's profile fields the agent uses."""
        with self._db() as db:
            learner = LearningTraceRepository(db).get_learner(student_id)
            if learner is None:
                raise LearnerNotFound(f"Learner {student_id!r} does not exist.")
            return {"student_id": learner.id, "display_name": learner.display_name,
                    "language": learner.preferred_language or "english",
                    "competency_level": learner.competency_level}

    # repository reads (B's AdaptationRepository)
    def get_latest_pretest(self, student_id: str, topic_id: str) -> dict[str, Any] | None:
        """Return the learner's latest pre-test for the topic, or None (has_pre_score + get_latest_assessment)."""
        with self._db() as db:
            repo = AdaptationRepository(db)
            if not repo.has_pre_score(student_id, topic_id):
                return None
            return repo.get_latest_assessment(student_id, topic_id, "pre")

    def get_last_qoe(self, student_id: str) -> float | None:
        """Return qoe_after of the learner's last experience, or None."""
        with self._db() as db:
            return AdaptationRepository(db).get_last_qoe(student_id)

    def get_experiences(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return stored experiences, newest first."""
        with self._db() as db:
            return AdaptationRepository(db).get_experiences(limit)

    def list_policies(self) -> list[dict[str, Any]]:
        """Return the predefined policies from the DB."""
        with self._db() as db:
            return AdaptationRepository(db).list_policies()

    def get_policy(self, policy_id: str) -> dict[str, Any] | None:
        """Return one policy row, or None."""
        with self._db() as db:
            return AdaptationRepository(db).get_policy(policy_id)

    def get_default_policy(self) -> dict[str, Any] | None:
        """Return the policy flagged is_default, or None."""
        with self._db() as db:
            return AdaptationRepository(db).get_default_policy()

    # pre-test / post-test / experience (B's services)
    def prepare_pretest(self, student_id: str, topic_id: str) -> list[dict[str, Any]]:
        """Return pre-test items without answers (generates the topic's question bank once)."""
        with self._db() as db:
            return b_pretest.prepare_pretest(student_id, topic_id, db=db,
                                             retriever=self.retriever, llm=self.mcq_llm)

    def submit_pretest(self, student_id: str, session_id: str, topic_id: str,
                       answers: Sequence[Any]) -> dict[str, Any]:
        """Score and store the pre-test; return {assessment_id, score, competency}."""
        with self._db() as db:
            return b_pretest.submit_pretest(student_id, session_id, topic_id, answers, db=db)

    def prepare_posttest(self, student_id: str, topic_id: str) -> list[dict[str, Any]]:
        """Return post-test items without answers (same bank as the pre-test)."""
        with self._db() as db:
            return b_posttest.prepare_posttest(student_id, topic_id, db=db)

    def submit_posttest_and_feedback(self, student_id: str, session_id: str, topic_id: str,
                                     answers: Sequence[Any], rating: int | None,
                                     comment: str | None) -> dict[str, Any]:
        """Score and store the post-test and feedback; return {score, competency, rating, comment, ...}."""
        with self._db() as db:
            return b_posttest.submit_posttest_and_feedback(student_id, session_id, topic_id, answers,
                                                           rating, comment, db=db)

    def store_experience(self, student_id: str, session_id: str, topic_id: str, policy_id: str,
                         context: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, Any]:
        """Persist the session's experience (idempotent per session)."""
        with self._db() as db:
            return b_experience.store_experience(student_id, session_id, topic_id, policy_id,
                                                 context, evaluation, db=db)


def make_app_gateway(*, retriever: Any, mcq_llm: Any) -> AppGateway:
    """Gateway over the real app DB (B's SessionLocal, data/app.db)."""
    from app.db.session import SessionLocal  # imported lazily: creates the app engine

    return AppGateway(SessionLocal, retriever=retriever, mcq_llm=mcq_llm)


def make_memory_gateway(*, retriever: Any, mcq_llm: Any) -> AppGateway:
    """Gateway over a private in-memory SQLite DB (tests, stub and mock runs)."""
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    gateway = AppGateway(sessionmaker(bind=engine, autoflush=False, future=True),
                         retriever=retriever, mcq_llm=mcq_llm)
    gateway.ensure_ready()
    return gateway


# ---- STUBS: Person B's Sunday metrics are not written yet ----------------------------------------

def evaluate_session_stub(qa_log: list[dict[str, Any]], rating: int | None, pre_score: float | None,
                          post_score: float | None, context: Mapping[str, Any],
                          runtime_config: Mapping[str, Any]) -> dict[str, float]:
    """STUB(B: metrics.qoe / metrics.sre / metrics.reward): placeholder evaluation in store_experience's shape.

    qoe_before = last_qoe from the context; qoe_after = rating/5 (0.5 without a rating);
    sre_before = 0.5; sre_after = 0.8 for local, 0.6 for cloud; reward = mean of qoe_after and sre_after.
    """
    qoe_after = round(rating / 5, 3) if rating else 0.5
    sre_after = 0.8 if runtime_config.get("route") == "local" else 0.6
    return {
        "qoe_before": float(context.get("last_qoe", 0.5)),
        "qoe_after": qoe_after,
        "sre_before": 0.5,
        "sre_after": sre_after,
        "reward": round(0.5 * qoe_after + 0.5 * sre_after, 3),
    }
