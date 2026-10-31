"""Answer matrix: every benchmark question × every variant, measured on the real network (resumable).

    .venv\\Scripts\\python -m evaluation.benchmarks.run_answer_matrix                 # all 150 questions, all variants
    .venv\\Scripts\\python -m evaluation.benchmarks.run_answer_matrix --per-topic 2  # a quick pilot

Rows are appended to evaluation/results/answer_matrix.csv as they finish; a re-run skips (question, variant) pairs
already there, so the Groq daily request cap or a crash only pauses the run. Rate-limited calls are retried after a
wait instead of being recorded (the router would otherwise fall back to the local model and mislabel the row).
"""
import argparse
import csv
import logging
import sys
import time

from evaluation.benchmarks.isolation import RESULTS, disable_tracing, isolate

isolate("answer_matrix", fresh_copy=False)

from agent.deps import get_services  # noqa: E402
from agent.knowledge import ensure_topic_index  # noqa: E402
from evaluation.benchmarks.harness import (  # noqa: E402
    TOPIC_IDS,
    VARIANTS,
    RateLimited,
    answer,
    load_doubts,
    load_questions,
)

OUT = RESULTS / "answer_matrix.csv"
DOUBT_VARIANTS = ["RAG-local", "P2", "RAG-cloud", "P4"]  # CRAG vs plain RAG on learner-style doubts
WAITS_S = (20, 40, 60, 90, 120, 180)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="mcq", choices=["mcq", "doubts"],
                        help="mcq: the 150 keyed items; doubts: 30 learner doubts + 10 off-topic (judged faithfulness)")
    parser.add_argument("--variants", nargs="+", default=None, choices=list(VARIANTS))
    parser.add_argument("--per-topic", type=int, default=None, help="questions per topic (default: all)")
    parser.add_argument("--kinds", nargs="+", default=["pre", "post"], choices=["pre", "post"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--wait-daily", action="store_true",
                        help="on Groq's daily token cap, wait and continue instead of stopping")
    args = parser.parse_args()
    args.variants = args.variants or (DOUBT_VARIANTS if args.set == "doubts" else list(VARIANTS))
    args.out = args.out or str(RESULTS / ("doubts_matrix.csv" if args.set == "doubts" else "answer_matrix.csv"))
    logging.basicConfig(level=logging.WARNING)
    disable_tracing()

    questions = load_doubts() if args.set == "doubts" else load_questions(tuple(args.kinds))
    if args.per_topic:
        questions = [q for t in TOPIC_IDS for q in [x for x in questions if x["topic_id"] == t][:args.per_topic]]
    done = set()
    try:
        with open(args.out, newline="", encoding="utf-8") as f:
            done = {(r["qid"], r["variant"]) for r in csv.DictReader(f)}
    except FileNotFoundError:
        pass
    todo = [(q, v) for q in questions for v in args.variants if (q["qid"], v) not in done]
    print(f"{len(questions)} questions × {len(args.variants)} variants: {len(done)} done, {len(todo)} to run",
          flush=True)

    services = get_services()
    for topic_id in {q["topic_id"] for q, _ in todo}:
        ensure_topic_index(services.crag.store, topic_id)
    services.router.clients["local"].warm_up()  # a cold model load must not count as answer latency
    record_idle_baseline()

    writer, handle = None, open(args.out, "a", newline="", encoding="utf-8")
    started = time.time()
    try:
        for n, (q, name) in enumerate(todo, 1):
            attempt = 0
            while True:
                try:
                    row = answer(q, VARIANTS[name], judge=args.set == "doubts")
                    break
                except RateLimited as exc:
                    if "per day" in str(exc) and args.wait_daily:  # the daily token cap frees up on a rolling window
                        print(f"  daily token cap reached, waiting 15 min ({time.strftime('%H:%M')})", flush=True)
                        time.sleep(900)
                        continue
                    if attempt >= len(WAITS_S):
                        print(f"giving up for now (rate limit / daily cap): {exc}", flush=True)
                        return 2
                    print(f"  rate limited, waiting {WAITS_S[attempt]}s: {str(exc)[:120]}", flush=True)
                    time.sleep(WAITS_S[attempt])
                    attempt += 1
            if writer is None:  # appending keeps the file's own columns (a newer harness may add fields)
                with open(args.out, newline="", encoding="utf-8") as f:
                    existing = next(csv.reader(f), None)
                writer = csv.DictWriter(handle, fieldnames=existing or list(row), extrasaction="ignore")
                if not existing:
                    writer.writeheader()
            writer.writerow(row)
            handle.flush()
            if n % 10 == 0 or n == len(todo):
                rate = n / max(time.time() - started, 1e-9) * 60
                print(f"{n}/{len(todo)}  ({rate:.1f}/min)  last: {q['qid']} {name} {row['provider']} "
                      f"correct={row['correct']} {row['total_ms']:.0f} ms", flush=True)
    finally:
        handle.close()
    return 0


def record_idle_baseline(seconds: float = 5.0) -> None:
    """Idle CPU/GPU load and GPU power, appended to idle_baseline.jsonl: energy is later reported above idle."""
    import json
    from datetime import datetime, timezone

    from evaluation.benchmarks.harness import ResourceSampler

    with ResourceSampler() as sampler:
        time.sleep(seconds)
    with open(RESULTS / "idle_baseline.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "seconds": round(sampler.elapsed_s, 2),
                            **sampler.summary()}) + "\n")


if __name__ == "__main__":
    sys.exit(main())
