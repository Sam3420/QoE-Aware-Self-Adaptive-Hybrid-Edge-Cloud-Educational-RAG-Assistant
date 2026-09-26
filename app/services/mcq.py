"""MCQ generation, topic question banks, and deterministic scoring.

Plain functions called by the agent's assessment nodes; nothing here imports LangGraph.
"""

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Protocol

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.repositories import AdaptationRepository
from app.domain.enums import CompetencyLevel
from app.domain.models import MCQItem
from app.llm.huggingface_provider import HuggingFaceInferenceProvider
from app.llm.provider import LLMProvider
from app.llm.types import LLMGenerateRequest, LLMGenerateResult, LLMProviderError
from app.services.errors import MCQGenerationError, VectorStoreUnavailableError

logger = get_logger(__name__)

ANSWER_LETTERS = "ABCD"

MCQ_SYSTEM_PROMPT = (
    "You write multiple-choice questions that check a learner's understanding of study material. "
    "Reply with JSON only: no prose, no markdown."
)


class KnowledgeRetriever(Protocol):
    """Retrieval contract of the project's knowledge retrieval service
    (``KnowledgeRetrievalService.retrieve_context`` on feature/knowledge-rag).

    Each returned hit is a dict with at least a ``content`` string.
    """

    def retrieve_context(
        self,
        *,
        resource_id: str,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:
        ...


@contextmanager
def session_scope(db: Session | None) -> Iterator[Session]:
    """Yield ``db`` unchanged, or open, commit, and close a new app session when it is None.

    A caller that passes its own session owns the transaction.
    """
    if db is not None:
        yield db
        return

    from app.db.session import SessionLocal  # imported lazily: creates the app engine

    with SessionLocal() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


def generate_mcqs(
    topic_id: str,
    n: int | None = None,
    *,
    retriever: KnowledgeRetriever | None = None,
    llm: LLMProvider | None = None,
) -> list[dict[str, Any]]:
    """Generate ``n`` MCQs for a topic from its retrieved chunks.

    Each item is ``{question, options[4], answer_index, concept}``. Invalid LLM output is
    retried ``settings.mcq_max_retries`` times, then MCQGenerationError is raised.
    Raises VectorStoreUnavailableError if no retriever is given.
    """
    items, _ = _generate_mcqs_with_result(topic_id, n, retriever=retriever, llm=llm)
    return items


def get_or_create_topic_items(
    topic_id: str,
    *,
    db: Session | None = None,
    retriever: KnowledgeRetriever | None = None,
    llm: LLMProvider | None = None,
) -> list[dict[str, Any]]:
    """Return the topic's shared MCQ items (answers included), generating and storing them once.

    The same items serve the pre-test and the post-test.
    """
    with session_scope(db) as session:
        repository = AdaptationRepository(session)
        bank = repository.get_question_bank(topic_id)
        if bank is not None:
            return bank["items"]

        items, result = _generate_mcqs_with_result(topic_id, None, retriever=retriever, llm=llm)
        repository.save_question_bank(
            topic_id=topic_id,
            items=items,
            model_provider=result.model_provider,
            model_name=result.model_name,
        )
        logger.info("question_bank_created topic_id=%s n_items=%s", topic_id, len(items))
        return items


def normalize_answer(answer: Any) -> int | None:
    """Map ``"A"``-``"D"`` (any case) or ``0``-``3`` to an option index; anything else is None."""
    if isinstance(answer, bool):
        return None
    if isinstance(answer, int):
        return answer if 0 <= answer <= 3 else None
    if isinstance(answer, str):
        letter = answer.strip().upper()
        if len(letter) == 1 and letter in ANSWER_LETTERS:
            return ANSWER_LETTERS.index(letter)
    return None


def score_mcqs(items: Sequence[dict[str, Any]], answers: Sequence[Any]) -> float:
    """Return the fraction of items answered correctly (0.0 for no items).

    Missing or invalid answers count as incorrect; answers beyond the item count are ignored.
    """
    if not items:
        return 0.0
    correct = sum(
        1
        for index, item in enumerate(items)
        if index < len(answers) and normalize_answer(answers[index]) == item["answer_index"]
    )
    return correct / len(items)


def competency_from_score(score: float, settings: Settings | None = None) -> str:
    """Map a 0-1 score to ``beginner`` / ``intermediate`` / ``advanced`` using config thresholds."""
    settings = settings or get_settings()
    if score < settings.competency_beginner_max:
        return CompetencyLevel.BEGINNER.value
    if score < settings.competency_intermediate_max:
        return CompetencyLevel.INTERMEDIATE.value
    return CompetencyLevel.ADVANCED.value


def _generate_mcqs_with_result(
    topic_id: str,
    n: int | None,
    *,
    retriever: KnowledgeRetriever | None,
    llm: LLMProvider | None,
) -> tuple[list[dict[str, Any]], LLMGenerateResult]:
    settings = get_settings()
    n = n or settings.mcq_n_items
    if retriever is None:
        raise VectorStoreUnavailableError(
            "No vector-store retriever is available for MCQ generation; "
            "pass the project's knowledge retriever (retrieve_context)."
        )

    hits = retriever.retrieve_context(
        resource_id=topic_id,
        query=f"key concepts of {topic_id}",
        top_k=settings.mcq_context_chunks,
    )
    chunks = [str(hit.get("content") or "").strip() for hit in hits]
    chunks = [chunk for chunk in chunks if chunk]
    if not chunks:
        raise MCQGenerationError(f"No indexed content was retrieved for topic {topic_id!r}.")

    llm = llm or _default_llm_provider(settings)
    base_prompt = _build_prompt(topic_id, chunks, n)
    prompt = base_prompt
    attempts = settings.mcq_max_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            result = llm.generate(
                LLMGenerateRequest(
                    prompt=prompt,
                    system_prompt=MCQ_SYSTEM_PROMPT,
                    model_id=settings.llm_model_id,
                    max_tokens=settings.mcq_max_tokens,
                    temperature=settings.llm_temperature,
                )
            )
        except LLMProviderError as exc:
            raise MCQGenerationError(f"LLM provider failed while generating MCQs: {exc}") from exc

        try:
            return _parse_items(result.text, n), result
        except ValueError as exc:
            logger.warning(
                "mcq_output_invalid topic_id=%s attempt=%s/%s reason=%s",
                topic_id,
                attempt,
                attempts,
                exc,
            )
            prompt = (
                f"{base_prompt}\n\nYour previous reply was rejected ({exc}). "
                "Reply again with only the JSON array."
            )

    raise MCQGenerationError(
        f"LLM returned invalid MCQ JSON for topic {topic_id!r} after {attempts} attempts."
    )


def _build_prompt(topic_id: str, chunks: list[str], n: int) -> str:
    context = "\n\n".join(f"[{index}] {chunk}" for index, chunk in enumerate(chunks, start=1))
    return (
        f"Topic: {topic_id}\n\nStudy material:\n{context}\n\n"
        f"Write exactly {n} multiple-choice questions answerable from the study material above. "
        "Return a JSON array where each element is an object with keys:\n"
        '  "question": string,\n'
        '  "options": array of exactly 4 distinct strings,\n'
        '  "answer_index": integer 0-3, the index of the correct option,\n'
        '  "concept": short string naming the concept tested.'
    )


def _parse_items(text: str, n: int) -> list[dict[str, Any]]:
    """Parse and validate the LLM reply; raise ValueError describing the first problem."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        cleaned = cleaned.rsplit("```", 1)[0]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON: {exc.msg}") from exc

    if not isinstance(data, list):
        raise ValueError("top-level JSON value is not an array")
    if len(data) != n:
        raise ValueError(f"expected {n} items, got {len(data)}")
    try:
        return [MCQItem.model_validate(item).model_dump() for item in data]
    except ValidationError as exc:
        raise ValueError(f"item failed validation: {exc.errors()[0]['msg']}") from exc


def _default_llm_provider(settings: Settings) -> LLMProvider:
    token = settings.hf_token.get_secret_value() if settings.hf_token else ""
    try:
        return HuggingFaceInferenceProvider(
            token=token,
            inference_url=settings.hf_inference_url,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    except LLMProviderError as exc:
        raise MCQGenerationError(f"LLM provider is not configured: {exc}") from exc
