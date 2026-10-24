"""Pre-build the formative-check pool (formative_items) so learners never wait for question generation in a session.

  .venv\\Scripts\\python scripts\\build_formative_pool.py --all                       # every topic with a vocabulary
  .venv\\Scripts\\python scripts\\build_formative_pool.py life_processes --per-concept 4
  .venv\\Scripts\\python scripts\\build_formative_pool.py --status                    # pool size per concept

Questions are generated with the cloud LLM only (Groq): the small local model writes wrong answer keys too often for
practice questions. On Groq's free-tier rate limit (8000 tokens/min for gpt-oss-120b) the script waits as asked.
Each topic's lesson-note index must exist (it is built the first time the topic is studied, or by this script).
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import formative_pool  # noqa: E402
import tracker  # noqa: E402
from agent.deps import get_services  # noqa: E402
from agent.interfaces import RouterLLMProvider  # noqa: E402
from agent.knowledge import ensure_topic_index  # noqa: E402
from knowledge_prep.catalog import TOPICS  # noqa: E402
from knowledge_prep.concepts import load_vocabulary  # noqa: E402


def topics_with_vocabulary(names: list[str], everything: bool) -> list[str]:
    candidates = [t.topic_id for t in TOPICS] if everything else names
    return [t for t in candidates if load_vocabulary(t) is not None]


def build(topic_id: str, per_concept: int, llm) -> None:
    services = get_services()
    services.app.ensure_ready()
    ensure_topic_index(services.crag.store, topic_id)  # the generator retrieves lesson context from it
    for concept in tracker.get_concepts(topic_id):
        with services.app._db() as db:
            have = formative_pool.pool_size(topic_id, concept.id, db=db)
        if have >= per_concept:
            print(f"  {concept.id:<36} {have} already")
            continue
        for attempt in range(6):
            try:
                with services.app._db() as db:
                    counts = formative_pool.top_up(topic_id, concept.id, per_concept - have, db=db, llm=llm)
                print(f"  {concept.id:<36} +{counts['added']} (duplicates {counts['duplicates']}, "
                      f"off-concept {counts['off_concept']}, LLM calls {counts['calls']})")
                break
            except Exception as exc:
                wait = re.search(r"try again in ([\d.]+)s", str(exc))
                if wait is None or attempt == 5:
                    print(f"  {concept.id:<36} FAILED: {type(exc).__name__}: {str(exc)[:200]}")
                    break
                seconds = float(wait.group(1)) + 1
                print(f"  {concept.id:<36} rate limited; waiting {seconds:.0f} s")
                time.sleep(seconds)


def status(topics: list[str]) -> None:
    services = get_services()
    services.app.ensure_ready()
    with services.app._db() as db:
        for topic_id in topics:
            print(topic_id)
            for concept in tracker.get_concepts(topic_id):
                print(f"  {concept.id:<36} {formative_pool.pool_size(topic_id, concept.id, db=db)}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Pre-build the formative-check question pool.")
    parser.add_argument("topics", nargs="*")
    parser.add_argument("--all", action="store_true", help="every catalogue topic with a vocabulary")
    parser.add_argument("--per-concept", type=int, default=4, help="questions to keep per concept (default 4)")
    parser.add_argument("--status", action="store_true", help="show pool sizes, generate nothing")
    args = parser.parse_args()
    topics = topics_with_vocabulary(args.topics, args.all or args.status and not args.topics)
    if not topics:
        parser.error("name topics that have a vocabulary, or use --all")
    if args.status:
        status(topics)
        return 0
    llm = RouterLLMProvider(get_services().router, providers=("cloud",))
    if not get_services().router.is_available("cloud"):
        raise SystemExit("No cloud LLM configured: set GROQ_API_KEY in .env.")
    for topic_id in topics:
        print(f"{topic_id}:")
        build(topic_id, args.per_concept, llm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
