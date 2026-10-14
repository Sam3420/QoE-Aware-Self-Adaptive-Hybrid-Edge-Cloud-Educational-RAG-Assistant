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
TOPUP_TEMPERATURE = 0.8

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


def extend_topic_items(
    topic_id: str,
    *,
    focus_concepts: Sequence[str] = (),
    db: Session | None = None,
    retriever: KnowledgeRetriever | None = None,
    llm: LLMProvider | None = None,
) -> int:
    """Generate new questions for an existing bank (focused on ``focus_concepts``) and append them.

    Questions already in the bank are dropped, and the bank never grows past MCQ_BANK_MAX_ITEMS.
    Returns how many items were added. Raises MCQGenerationError like the first generation.
    """
    settings = get_settings()
    with session_scope(db) as session:
        repository = AdaptationRepository(session)
        bank = repository.get_question_bank(topic_id)
        if bank is None:
            raise MCQGenerationError(f"Topic {topic_id!r} has no question bank to extend.")
        room = settings.mcq_bank_max_items - len(bank["items"])
        if room <= 0:
            return 0
        existing = [item["question"] for item in bank["items"]]
        seen = {question_key(q) for q in existing}
        fresh: list[dict[str, Any]] = []
        # Small models tend to repeat the bank verbatim at low temperature; sample more freely and,
        # if every question was a duplicate, try once more with the next window of study material.
        for attempt in range(settings.mcq_max_retries + 1):
            items, _ = _generate_mcqs_with_result(
                topic_id, None, retriever=retriever, llm=llm, focus_concepts=focus_concepts,
                avoid_questions=existing, temperature=max(settings.llm_temperature, TOPUP_TEMPERATURE),
                context_offset=len(existing) + attempt * settings.mcq_context_chunks)
            for item in items:
                key = question_key(item["question"])
                if key not in seen:
                    seen.add(key)
                    fresh.append(item)
            if fresh:
                break
        fresh = fresh[:room]
        if fresh:
            repository.append_question_bank_items(topic_id, fresh)
        logger.info("question_bank_extended topic_id=%s added=%s focus=%s", topic_id, len(fresh), list(focus_concepts))
        return len(fresh)


EXPLAIN_SYSTEM_PROMPT = (
    "You explain the answers to multiple-choice questions for a school learner, using only the study "
    "material given. Reply with JSON only: no prose, no markdown."
)


def explain_items(
    topic_id: str,
    item_ids: Sequence[int],
    *,
    db: Session | None = None,
    retriever: KnowledgeRetriever | None = None,
    llm: LLMProvider | None = None,
) -> int:
    """Write an explanation for each bank item in ``item_ids`` that has none, and store it in the bank.

    Items generated with an explanation are left alone. One LLM call covers all missing items; it is
    grounded in study material retrieved for those questions. Returns how many explanations were added.
    Raises MCQGenerationError if the LLM fails or replies with invalid JSON after the retries.
    """
    settings = get_settings()
    with session_scope(db) as session:
        repository = AdaptationRepository(session)
        bank = repository.get_question_bank(topic_id)
        if bank is None:
            raise MCQGenerationError(f"Topic {topic_id!r} has no question bank to explain.")
        missing = [i for i in dict.fromkeys(item_ids)
                   if 0 <= i < len(bank["items"]) and not str(bank["items"][i].get("explanation") or "").strip()]
        if not missing:
            return 0
        if retriever is None:
            raise VectorStoreUnavailableError("No vector-store retriever is available to ground explanations.")
        items = [bank["items"][i] for i in missing]
        hits = retriever.retrieve_context(resource_id=topic_id, query=" ".join(item["question"] for item in items),
                                          top_k=settings.mcq_context_chunks)
        chunks = [c for c in (str(h.get("content") or "").strip() for h in hits) if c]
        llm = llm or _default_llm_provider(settings)
        prompt = _build_explain_prompt(topic_id.replace("_", " "), chunks, items)
        schema = {"type": "array", "items": {"type": "string"}, "minItems": len(items), "maxItems": len(items)}
        attempts = settings.mcq_max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                result = llm.generate(LLMGenerateRequest(
                    prompt=prompt, system_prompt=EXPLAIN_SYSTEM_PROMPT, model_id=settings.llm_model_id,
                    max_tokens=settings.mcq_max_tokens, temperature=settings.llm_temperature, json_schema=schema))
            except LLMProviderError as exc:
                raise MCQGenerationError(f"LLM provider failed while writing explanations: {exc}") from exc
            try:
                texts = _parse_explanations(result.text, len(items))
                break
            except ValueError as exc:
                logger.warning("explanations_invalid topic_id=%s attempt=%s/%s reason=%s",
                               topic_id, attempt, attempts, exc)
        else:
            raise MCQGenerationError(f"LLM returned invalid explanations for topic {topic_id!r} "
                                     f"after {attempts} attempts.")
        repository.set_question_bank_explanations(topic_id, dict(zip(missing, texts)))
        logger.info("question_bank_explained topic_id=%s added=%s", topic_id, len(texts))
        return len(texts)


def _build_explain_prompt(topic: str, chunks: list[str], items: Sequence[dict[str, Any]]) -> str:
    context = "\n\n".join(f"[{index}] {chunk}" for index, chunk in enumerate(chunks, start=1)) or "(none retrieved)"
    questions = "\n\n".join(
        f"Question {n}: {item['question']}\n"
        + "\n".join(f"  {ANSWER_LETTERS[i]}. {option}" for i, option in enumerate(item["options"]))
        + f"\nCorrect answer: {ANSWER_LETTERS[item['answer_index']]}. {item['options'][item['answer_index']]}"
        for n, item in enumerate(items, start=1))
    return (
        f"Topic: {topic}\n\nStudy material:\n{context}\n\n{questions}\n\n"
        f"For each of the {len(items)} questions, in order, write one or two sentences explaining why the "
        "correct answer is right, using the study material. Return a JSON array of "
        f"{len(items)} strings."
    )


def _parse_explanations(text: str, n: int) -> list[str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        cleaned = cleaned.rsplit("```", 1)[0]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON: {exc.msg}") from exc
    if not isinstance(data, list) or len(data) != n:
        raise ValueError(f"expected an array of {n} strings")
    texts = [str(t).strip() if isinstance(t, (str, int, float)) else "" for t in data]
    if not all(texts):
        raise ValueError("an explanation is empty")
    return texts


def question_key(question: str) -> str:
    """Case- and whitespace-insensitive identity of a question."""
    return " ".join(question.lower().split())


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
    focus_concepts: Sequence[str] = (),
    avoid_questions: Sequence[str] = (),
    temperature: float | None = None,
    context_offset: int = 0,
) -> tuple[list[dict[str, Any]], LLMGenerateResult]:
    settings = get_settings()
    n = n or settings.mcq_n_items
    if retriever is None:
        raise VectorStoreUnavailableError(
            "No vector-store retriever is available for MCQ generation; "
            "pass the project's knowledge retriever (retrieve_context)."
        )

    topic_text = topic_id.replace("_", " ")
    query = f"{topic_text}: {', '.join(focus_concepts)}" if focus_concepts else f"key concepts of {topic_text}"
    # A top-up (context_offset > 0) fetches a wider pool and rotates through it, so the model sees
    # different study material than it did for the questions already in the bank.
    pool = settings.mcq_context_chunks * (2 if context_offset else 1)
    hits = retriever.retrieve_context(
        resource_id=topic_id,
        query=query,
        top_k=pool,
    )
    chunks = [str(hit.get("content") or "").strip() for hit in hits]
    chunks = [chunk for chunk in chunks if chunk]
    if context_offset and len(chunks) > settings.mcq_context_chunks:
        start = context_offset % len(chunks)
        chunks = [*chunks[start:], *chunks[:start]][: settings.mcq_context_chunks]
    if not chunks:
        raise MCQGenerationError(f"No indexed content was retrieved for topic {topic_id!r}.")

    llm = llm or _default_llm_provider(settings)
    base_prompt = _build_prompt(topic_text, chunks, n, focus_concepts, avoid_questions)
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
                    temperature=settings.llm_temperature if temperature is None else temperature,
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


def _build_prompt(topic: str, chunks: list[str], n: int, focus_concepts: Sequence[str] = (),
                  avoid_questions: Sequence[str] = ()) -> str:
    context = "\n\n".join(f"[{index}] {chunk}" for index, chunk in enumerate(chunks, start=1))
    focus = (f"Focus on these concepts the learner found difficult: {', '.join(focus_concepts)}.\n"
             if focus_concepts else "")
    # Only the most recent questions are listed, to keep the prompt within small models' context.
    avoid = ("Do not repeat or rephrase any of these existing questions:\n"
             + "\n".join(f"- {q}" for q in list(avoid_questions)[-20:]) + "\n" if avoid_questions else "")
    return (
        f"Topic: {topic}\n\nStudy material:\n{context}\n\n"
        f"Write exactly {n} multiple-choice questions answerable from the study material above.\n"
        f"{focus}{avoid}"
        "Return a JSON array where each element is an object with keys:\n"
        '  "question": string,\n'
        '  "options": array of exactly 4 distinct strings,\n'
        '  "answer_index": integer 0-3, the index of the correct option,\n'
        '  "concept": the concept tested as a short name of 1-4 words, e.g. "nitrogen fixation" '
        '(not a sentence),\n'
        '  "explanation": one or two sentences, based on the study material, saying why the correct '
        'option is right.'
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
        items = [MCQItem.model_validate(item).model_dump() for item in data]
    except ValidationError as exc:
        raise ValueError(f"item failed validation: {exc.errors()[0]['msg']}") from exc
    for item in items:  # no explanation given: leave the key out so explain_items can fill it later
        if not item["explanation"].strip():
            del item["explanation"]
    return items


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
