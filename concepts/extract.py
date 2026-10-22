"""Concept vocabulary + question-to-concept map for a topic (knowledge_prep/concepts/<topic_id>.json).

    .venv\\Scripts\\python -m concepts.extract life_processes heredity      # these topics
    .venv\\Scripts\\python -m concepts.extract --all                         # every catalogue topic
    .venv\\Scripts\\python -m concepts.extract heredity --remap              # redo the map of a draft vocabulary

Per topic:
1. Vocabulary: if the topic has none, Groq drafts 6-12 concepts from the lesson notes plus the question bank's labels
   (scripts/extract_concepts.py: validated, evidence quoted from the notes) and it is saved with status "draft".
   An existing vocabulary's concepts are never changed here.
2. Map: Groq assigns every question in the topic's bank (app DB) exactly one concept, with a confidence. A mapping
   needs review when the confidence is below REVIEW_BELOW or it disagrees with the concept the item's own label names.
   Stored as ``question_concept_map`` (key: bank position) and as each concept's ``source_question_ids``.

Re-runnable: an approved vocabulary is never rewritten once it has a map (one without a map gets the map added, its
concepts and status unchanged); a draft keeps its map unless --remap. Review the flagged mappings in the JSON by hand.
"""
import argparse
import json
import sys
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from knowledge_prep.catalog import TOPICS, topic_title
from knowledge_prep.concepts import ConceptMatcher, QuestionMapping, Vocabulary, load_vocabulary, save_vocabulary
from services.llm_clients import GenerationRequest, LLMError

REVIEW_BELOW = 0.7  # LLM confidence under this -> needs_review

SYSTEM = ("You are a curriculum designer for NCERT school science. You decide which concept each test question "
          "assesses. Reply with one JSON object and nothing else.")


class _Mapped(BaseModel):
    question_id: int
    concept_id: str
    confidence: float = Field(ge=0.0, le=1.0)


def bank_items(topic_id: str) -> list[dict[str, Any]]:
    """The topic's question-bank items in bank order (empty when it has no bank)."""
    from app.db.repositories import AdaptationRepository
    from app.db.session import SessionLocal

    with SessionLocal() as db:
        bank = AdaptationRepository(db).get_question_bank(topic_id)
    return list((bank or {}).get("items") or [])


def mapping_prompt(vocabulary: Vocabulary, items: list[dict[str, Any]], feedback: str = "") -> str:
    concepts = "\n".join(f"- {c.id}: {c.name}. {c.description}" + (f" (also: {', '.join(c.aliases)})" if c.aliases
                                                                      else "") for c in vocabulary.concepts)
    questions = "\n".join(
        f"[{i}] {item['question']} | options: {' / '.join(map(str, item.get('options') or []))}"
        f" | correct: {_correct(item)} | label: {item.get('concept') or '-'}" for i, item in enumerate(items))
    return (
        f"Topic: {topic_title(vocabulary.topic_id)}\n\nConcepts:\n{concepts}\n\nQuestions:\n{questions}\n\n"
        "Assign every question exactly one concept id from the list: the concept a learner must understand to answer "
        "it correctly. The label was written when the question was generated and may be vague or wrong. "
        "Give your confidence from 0 to 1 (below 0.7 when two concepts fit or none fits well).\n"
        'Return JSON: {"mappings": [{"question_id": 0, "concept_id": "an_id_from_the_list", "confidence": 0.9}]}'
        + (f"\n\nYour previous reply was rejected: {feedback}. Fix that and reply again." if feedback else ""))


def map_questions(vocabulary: Vocabulary, items: list[dict[str, Any]], client: Any,
                  attempts: int = 3) -> dict[str, QuestionMapping]:
    """Map every bank item to one concept with the LLM; retries with the problem when a reply is invalid."""
    from scripts.extract_concepts import _generate_waiting_on_rate_limit

    ids = {c.id for c in vocabulary.concepts}
    labels = ConceptMatcher(vocabulary)  # exact name/id/alias only (no embedder): a cross-check, not the mapping
    feedback = ""
    for attempt in range(1, attempts + 1):
        request = GenerationRequest(system=SYSTEM, prompt=mapping_prompt(vocabulary, items, feedback),
                                    max_tokens=3000, temperature=0.0, operation="map-questions-to-concepts")
        try:
            text = _generate_waiting_on_rate_limit(client, request)
            rows = [_Mapped.model_validate(r) for r in json.loads(text[text.index("{"):text.rindex("}") + 1])["mappings"]]
            got = sorted(r.question_id for r in rows)
            if got != list(range(len(items))):
                raise ValueError(f"need each question_id 0..{len(items) - 1} exactly once, got {got}")
            unknown = sorted({r.concept_id for r in rows} - ids)
            if unknown:
                raise ValueError(f"unknown concept ids {unknown}; use only ids from the list")
        except (LLMError, ValueError, KeyError, ValidationError) as exc:
            feedback = f"{type(exc).__name__}: {str(exc)[:300]}"
            print(f"  map attempt {attempt}: rejected ({feedback})")
            continue
        out = {}
        for r in sorted(rows, key=lambda r: r.question_id):
            item = items[r.question_id]
            notes = []
            if r.confidence < REVIEW_BELOW:
                notes.append(f"low confidence {r.confidence:.2f}")
            named = labels.match(str(item.get("concept") or "")).concept_id
            if named is not None and named != r.concept_id:
                notes.append(f"the item's label names {named}")
            out[str(r.question_id)] = QuestionMapping(concept_id=r.concept_id, question=item["question"],
                                                      confidence=r.confidence, needs_review=bool(notes),
                                                      note="; ".join(notes))
        return out
    raise RuntimeError(f"could not map {vocabulary.topic_id}'s questions: {feedback}")


def with_map(vocabulary: Vocabulary, mapping: dict[str, QuestionMapping]) -> Vocabulary:
    """The vocabulary with this map and each concept's source_question_ids (concepts and status unchanged)."""
    by_concept: dict[str, list[int]] = {c.id: [] for c in vocabulary.concepts}
    for question_id, m in mapping.items():
        by_concept[m.concept_id].append(int(question_id))
    concepts = [c.model_copy(update={"source_question_ids": sorted(by_concept[c.id])}) for c in vocabulary.concepts]
    return Vocabulary.model_validate({**vocabulary.model_dump(), "concepts": [c.model_dump() for c in concepts],
                                      "question_concept_map": {k: v.model_dump() for k, v in mapping.items()}})


def extract(topic_id: str, *, remap: bool = False, client: Any = None) -> Vocabulary | None:
    """Draft the vocabulary if missing, then add the question map; returns what is saved (None: nothing to do)."""
    vocabulary = load_vocabulary(topic_id)
    if vocabulary is None:
        from scripts.extract_concepts import draft

        print(f"{topic_id}: drafting the vocabulary")
        vocabulary = draft(topic_id)
        save_vocabulary(vocabulary)
    if vocabulary.question_concept_map and (vocabulary.status == "approved" or not remap):
        print(f"{topic_id}: [{vocabulary.status}] already has a question map; left unchanged"
              + ("" if vocabulary.status == "approved" else " (use --remap to redo it)"))
        return None
    items = bank_items(topic_id)
    if not items:
        print(f"{topic_id}: no question bank in the app DB yet; no map (run again once the topic has questions)")
        return vocabulary
    if client is None:
        from agent.deps import make_cloud_client

        client = make_cloud_client()
        if not client.is_available():
            raise SystemExit("No cloud LLM configured: set GROQ_API_KEY in .env.")
    vocabulary = with_map(vocabulary, map_questions(vocabulary, items, client))
    save_vocabulary(vocabulary)
    report(vocabulary)
    return vocabulary


def report(vocabulary: Vocabulary) -> None:
    flagged = {k: m for k, m in vocabulary.question_concept_map.items() if m.needs_review}
    print(f"{vocabulary.topic_id}: [{vocabulary.status}] {len(vocabulary.question_concept_map)} questions mapped, "
          f"{len(flagged)} need review")
    for question_id, m in flagged.items():
        print(f"  q{question_id} -> {m.concept_id} ({m.note}): {m.question[:90]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("topics", nargs="*", help="topic ids (default with --all: every catalogue topic)")
    parser.add_argument("--all", action="store_true", help="every catalogue topic")
    parser.add_argument("--remap", action="store_true", help="redo the map of draft vocabularies")
    args = parser.parse_args()
    topics = [t.topic_id for t in TOPICS] if args.all else args.topics
    if not topics:
        parser.error("name topic ids or use --all")
    known = {t.topic_id for t in TOPICS}
    if unknown := [t for t in topics if t not in known]:
        parser.error(f"unknown topics {unknown}; catalogue: {sorted(known)}")
    for topic_id in topics:
        extract(topic_id, remap=args.remap)
    return 0


def _correct(item: dict[str, Any]) -> str:
    options, index = item.get("options") or [], item.get("answer_index")
    return str(options[index]) if isinstance(index, int) and 0 <= index < len(options) else "?"


if __name__ == "__main__":
    sys.exit(main())
