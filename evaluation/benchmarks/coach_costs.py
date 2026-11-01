"""Result 9: what each proactive-coach step costs (time, LLM calls, network), measured on the real components.

    .venv\\Scripts\\python -m evaluation.benchmarks.coach_costs              # local steps + scaffold experiment
    .venv\\Scripts\\python -m evaluation.benchmarks.coach_costs --no-llm     # local steps only (no API use)

1. Steps without a model (repeated timings): session planning (missed concepts + goal rules over the stored
   pre-tests), the coach's rule decision, picking formative items from the pool, loading the parallel post-test.
2. Answer scaffolding (coach_mode=on shapes answers): the 30 learner doubts answered by P1 (edge) and P3 (cloud)
   plain, step-by-step, and step-by-step with a hint first: latency, output tokens, answer length.
3. "Another video" costs a recommendation run: measured by bench_videos.py (concept-focused queries).

Writes evaluation/results/coach/{local_steps.csv, scaffold.csv}.
"""
import argparse
import json
import sqlite3
import statistics
import sys
import time

from evaluation.benchmarks.isolation import RESULTS, disable_tracing, isolate

isolate("coach_costs", fresh_copy=True)

import pandas as pd  # noqa: E402

from agent import coach_data  # noqa: E402
from agent.coach import choose_action  # noqa: E402
from agent.deps import get_services  # noqa: E402
from agent.nodes.coaching import load_vocabulary  # noqa: E402
from agent.planning import missed_concepts, plan_goals  # noqa: E402
from app.core.config import get_settings as app_settings  # noqa: E402
from evaluation.benchmarks.harness import TOPIC_IDS, VARIANTS, RateLimited, answer, load_doubts  # noqa: E402

OUT = RESULTS / "coach"
SCAFFOLDS = {"none": None, "step_by_step": {"style": "step_by_step", "support": "direct"},
             "step_by_step+hint_first": {"style": "step_by_step", "support": "hint_first"}}


def timed(fn, reps: int) -> list[float]:
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return out


def local_steps(reps: int = 30) -> pd.DataFrame:
    services = get_services()
    embedder = getattr(services.crag.store, "embedder", None)
    with sqlite3.connect(app_settings().database_path) as con:
        pretests = [(t, {"items": json.loads(i), "answers": json.loads(a)}) for t, i, a in con.execute(
            "SELECT topic_id, items, answers FROM assessments WHERE type = 'pre'") if t in TOPIC_IDS]
        posts = {t: [json.loads(i) for (i,) in con.execute("SELECT item FROM parallel_test_items WHERE topic_id = ?",
                                                           (t,))] for t in TOPIC_IDS}
    rows = []

    def add(step: str, samples: list[float], note: str) -> None:
        rows.append({"step": step, "n": len(samples), "median_ms": statistics.median(samples),
                     "p95_ms": sorted(samples)[int(0.95 * (len(samples) - 1))], "llm_calls": 0, "network_calls": 0,
                     "note": note})

    plan_times = []
    for topic_id, assessment in pretests:
        vocab = load_vocabulary(topic_id)
        mastery = coach_data.get_mastery("coach-cost-probe", topic_id)

        def plan() -> None:
            missed = missed_concepts(vocab, assessment, embedder)
            plan_goals(vocab, missed, mastery)

        plan_times += timed(plan, reps)
    if plan_times:
        add("plan session goals", plan_times, f"{len(pretests)} stored pre-tests × {reps}")

    decide = []
    for topic_id in TOPIC_IDS:
        vocab = load_vocabulary(topic_id)
        goals = [{"concept_id": c.id, "name": c.name} for c in vocab.concepts[:3]]
        snapshot = {c.id: {"mastery": 0.4, "attempts": 2, "correct": 1, "covered": True, "asked": 1}
                    for c in vocab.concepts}
        decide += timed(lambda: choose_action(goals, snapshot, last_concept=goals[0]["concept_id"]), reps * 10)
    add("coach rule decision", decide, "choose_action, 3 goals")

    formative = []
    for topic_id in TOPIC_IDS:
        for concept in load_vocabulary(topic_id).concepts:
            formative += timed(lambda: coach_data.get_formative_items(topic_id, concept.id, 2, []), 3)
    add("pick 2 formative items", formative, "pre-built pool, all concepts")

    post = []
    for topic_id, items in posts.items():
        post += timed(lambda: [i["answer_index"] == 0 for i in items], reps)
    add("score parallel post-test", post, "15 items")
    return pd.DataFrame(rows)


def scaffold_experiment(policies: list[str]) -> pd.DataFrame:
    doubts = [d for d in load_doubts() if d["kind"] == "doubt"]
    services = get_services()
    services.router.clients["local"].warm_up()
    rows = []
    for q in doubts:
        for pid in policies:
            for name, scaffold in SCAFFOLDS.items():
                attempt = 0
                while True:
                    try:
                        row = answer(q, VARIANTS[pid], scaffold=scaffold)
                        break
                    except RateLimited as exc:
                        if "per day" in str(exc):  # Groq's daily token cap: wait for the rolling window
                            print(f"  daily token cap, waiting 15 min ({time.strftime('%H:%M')})", flush=True)
                            time.sleep(900)
                            continue
                        wait = (20, 40, 60, 120, 180)[min(attempt, 4)]
                        attempt += 1
                        print(f"  rate limited, waiting {wait}s: {str(exc)[:100]}", flush=True)
                        time.sleep(wait)
                rows.append({**{k: row[k] for k in ("qid", "topic_id", "variant", "provider", "total_ms", "llm_ms",
                                                     "completion_tokens", "n_cloud_calls")},
                             "scaffold": name, "answer_chars": len(row["response"]), "response": row["response"]})
        print(f"  {q['qid']} done", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--policies", nargs="+", default=["P1", "P3"])
    args = parser.parse_args()
    disable_tracing()
    OUT.mkdir(parents=True, exist_ok=True)
    steps = local_steps()
    steps.to_csv(OUT / "local_steps.csv", index=False)
    print(steps.to_string(index=False))
    existing = OUT / "scaffold.csv"
    if not args.no_llm and existing.exists():
        done = pd.read_csv(existing)
        if set(args.policies) <= set(done.variant) and len(done) >= 30 * len(SCAFFOLDS) * len(set(done.variant)):
            print(f"scaffold experiment already complete in {existing} ({len(done)} rows): not re-run")
            return 0
    if not args.no_llm:
        scaffold = scaffold_experiment(args.policies)
        scaffold.to_csv(OUT / "scaffold.csv", index=False)
        print(scaffold.groupby(["variant", "scaffold"])[["total_ms", "completion_tokens", "answer_chars"]].mean())
    return 0


if __name__ == "__main__":
    sys.exit(main())
