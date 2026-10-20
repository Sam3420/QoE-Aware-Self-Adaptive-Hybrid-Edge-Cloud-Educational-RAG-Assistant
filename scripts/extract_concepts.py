"""Draft, show and approve topic concept vocabularies (knowledge_prep/concepts/<topic_id>.json).

  .venv\\Scripts\\python scripts\\extract_concepts.py --all                  # draft every catalogue topic without one
  .venv\\Scripts\\python scripts\\extract_concepts.py life_processes --force  # redraft one topic
  .venv\\Scripts\\python scripts\\extract_concepts.py --show life_processes   # print it for review
  .venv\\Scripts\\python scripts\\extract_concepts.py --approve life_processes --by "Reviewer Name"

Drafting uses the cloud LLM (Groq, GROQ_API_KEY) on the topic's lesson notes plus the concept labels already used by
the topic's question bank. A draft is accepted only if it validates: 6-12 snake_case concepts, unique names, and an
evidence sentence for each that appears verbatim in the notes. Review the draft, edit the JSON if needed, then
approve it; the tracker reports whether the vocabulary it used was approved.
"""
import argparse
import json
import re
import time
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError  # noqa: E402

from agent.deps import make_cloud_client  # noqa: E402
from config import settings  # noqa: E402
from knowledge_prep.catalog import TOPICS, topic_title  # noqa: E402
from knowledge_prep.concepts import (  # noqa: E402
    MAX_CONCEPTS, MIN_CONCEPTS, Vocabulary, load_vocabulary, path_for, save_vocabulary, vocabulary_problems)
from services.llm_clients import GenerationRequest, LLMError  # noqa: E402

SYSTEM = ("You are a curriculum designer for NCERT school science. You list the distinct concepts a lesson teaches. "
          "Reply with one JSON object and nothing else.")


def bank_labels(topic_id: str) -> list[str]:
    """Concept labels of the topic's question bank in the app DB (empty when there is none)."""
    try:
        from app.db.repositories import AdaptationRepository
        from app.db.session import SessionLocal

        with SessionLocal() as db:
            bank = AdaptationRepository(db).get_question_bank(topic_id)
        return sorted({str(i.get("concept", "")).strip() for i in (bank or {}).get("items", []) if i.get("concept")})
    except Exception:  # no DB yet: the notes alone are enough
        return []


def prompt(topic_id: str, notes: str, labels: list[str], feedback: str = "") -> str:
    hint = ("Questions on this topic already use these concept labels; make sure each maps onto one of your concepts "
            f"(by name or alias): {'; '.join(labels)}\n\n" if labels else "")
    return (
        f"Topic: {topic_title(topic_id)} (NCERT Class 10 Biology)\n\nLesson notes:\n{notes}\n\n{hint}"
        f"List the {MIN_CONCEPTS}-{MAX_CONCEPTS} distinct concepts these notes teach that a learner can be tested on. "
        "Each concept must be specific enough for a few multiple-choice questions and must not overlap another. "
        "Together the concepts must cover every main section of the notes; merge small related points rather than "
        "leaving a section out. "
        'Return JSON: {"concepts": [{"id": "snake_case_id", "name": "short name, 1-4 words", '
        '"description": "one sentence on what the learner should understand", '
        '"aliases": ["other names or key terms for it"], '
        '"evidence": "one sentence copied exactly, word for word, from the notes that teaches it"}]}'
        + (f"\n\nYour previous reply was rejected: {feedback}. Fix that and reply again." if feedback else "")
    )


def draft(topic_id: str) -> Vocabulary:
    notes_path = settings.knowledge_samples_dir / f"{topic_id}.txt"
    notes = notes_path.read_text(encoding="utf-8")
    labels = bank_labels(topic_id)
    client = make_cloud_client()
    if not client.is_available():
        raise SystemExit("No cloud LLM configured: set GROQ_API_KEY in .env.")
    feedback = ""
    for attempt in (1, 2, 3):
        request = GenerationRequest(system=SYSTEM, prompt=prompt(topic_id, notes, labels, feedback), max_tokens=4000,
                                    temperature=0.2, operation="extract-concepts")
        try:
            text = _generate_waiting_on_rate_limit(client, request)
            data = json.loads(text[text.index("{"):text.rindex("}") + 1])
            vocab = Vocabulary(topic_id=topic_id, generated_by=f"{client.label}:{client.model}",
                               generated_at=datetime.now(timezone.utc), concepts=data["concepts"])
        except (LLMError, ValueError, KeyError, ValidationError) as exc:
            feedback = f"{type(exc).__name__}: {str(exc)[:300]}"
            print(f"  attempt {attempt}: rejected ({feedback})")
            continue
        problems = vocabulary_problems(vocab, notes)
        if not problems:
            return vocab
        feedback = ("; ".join(problems) + " (copy each evidence sentence exactly from the notes, and give every "
                    "concept names and aliases no other concept uses)")
        print(f"  attempt {attempt}: rejected ({feedback})")
    raise SystemExit(f"Could not draft a valid vocabulary for {topic_id}: {feedback}")


def _generate_waiting_on_rate_limit(client, request, max_waits: int = 5) -> str:
    """Generate; on a rate-limit reply (HTTP 429, e.g. Groq's free tier: 8000 tokens/min for gpt-oss-120b) wait the
    time the API asks for and try again. A draft is ~3000 tokens, so several topics in a row hit the limit."""
    for _ in range(max_waits):
        try:
            return client.generate(request).text
        except LLMError as exc:
            match = re.search(r"try again in ([\d.]+)s", str(exc))
            if "HTTP 429" not in str(exc) or match is None:
                raise
            wait = float(match.group(1)) + 1.0
            print(f"  rate limited; waiting {wait:.0f} s")
            time.sleep(wait)
    return client.generate(request).text


def show(topic_id: str) -> None:
    vocab = load_vocabulary(topic_id)
    if vocab is None:
        raise SystemExit(f"No vocabulary for {topic_id}; draft it first.")
    reviewed = f", reviewed by {vocab.reviewed_by}" if vocab.reviewed_by else ""
    print(f"{topic_title(topic_id)} [{vocab.status}{reviewed}] {len(vocab.concepts)} concepts, {vocab.generated_by}")
    notes = (settings.knowledge_samples_dir / f"{topic_id}.txt").read_text(encoding="utf-8")
    for problem in vocabulary_problems(vocab, notes):
        print(f"  PROBLEM: {problem}")
    for c in vocab.concepts:
        print(f"\n- {c.id}: {c.name}\n    {c.description}\n    aliases: {', '.join(c.aliases) or '-'}"
              f"\n    evidence: \"{c.evidence}\"")


def approve(topic_id: str, reviewer: str) -> None:
    vocab = load_vocabulary(topic_id)
    if vocab is None:
        raise SystemExit(f"No vocabulary for {topic_id}; draft it first.")
    notes = (settings.knowledge_samples_dir / f"{topic_id}.txt").read_text(encoding="utf-8")
    if problems := vocabulary_problems(vocab, notes):
        raise SystemExit("Not approved; fix these first: " + "; ".join(problems))
    vocab = vocab.model_copy(update={"status": "approved", "reviewed_by": reviewer,
                                     "reviewed_at": datetime.now(timezone.utc)})
    save_vocabulary(Vocabulary.model_validate(vocab.model_dump()))
    print(f"Approved {topic_id} ({len(vocab.concepts)} concepts) by {reviewer}.")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Draft, show and approve topic concept vocabularies.")
    parser.add_argument("topics", nargs="*", help="topic ids to draft")
    parser.add_argument("--all", action="store_true", help="draft every catalogue topic")
    parser.add_argument("--force", action="store_true", help="redraft even if a vocabulary exists (drops approval)")
    parser.add_argument("--show", metavar="TOPIC")
    parser.add_argument("--approve", metavar="TOPIC")
    parser.add_argument("--by", help="reviewer name for --approve")
    args = parser.parse_args()
    if args.show:
        show(args.show)
        return 0
    if args.approve:
        if not args.by:
            raise SystemExit("--approve needs --by \"Reviewer Name\"")
        approve(args.approve, args.by)
        return 0
    topics = [t.topic_id for t in TOPICS] if args.all else args.topics
    if not topics:
        parser.error("name topics or use --all")
    for topic_id in topics:
        if path_for(topic_id).exists() and not args.force:
            print(f"{topic_id}: already drafted ({load_vocabulary(topic_id).status}); use --force to redraft")
            continue
        print(f"{topic_id}: drafting…")
        vocab = draft(topic_id)
        print(f"  saved {save_vocabulary(vocab)} ({len(vocab.concepts)} concepts, status draft)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
