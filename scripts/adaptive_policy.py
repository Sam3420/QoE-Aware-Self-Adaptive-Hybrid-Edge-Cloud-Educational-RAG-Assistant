"""Command-line tools for the adaptive policy module (run from the repository root).

  .venv\\Scripts\\python scripts\\adaptive_policy.py init              # create tables, seed policies, sync FAISS
  .venv\\Scripts\\python scripts\\adaptive_policy.py status            # experiences, index, weights in use
  .venv\\Scripts\\python scripts\\adaptive_policy.py select --rtt 300 --cpu 0.4 --mem 0.6 --competency 0.5
  .venv\\Scripts\\python scripts\\adaptive_policy.py estimate-weights --group eq [--activate]
  .venv\\Scripts\\python scripts\\adaptive_policy.py estimate-weights --all
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.deps import get_services  # noqa: E402
from agent.policy_engine import PolicyEngine  # noqa: E402
from agent.weight_estimation import estimate_group  # noqa: E402
from app.db.repositories import AdaptiveLearningRepository  # noqa: E402
from config import settings  # noqa: E402
from metrics.weights import GROUP_KEYS  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("status")
    sel = sub.add_parser("select", help="dry-run a policy decision for a context (nothing is executed)")
    sel.add_argument("--rtt", type=float, default=200.0)
    sel.add_argument("--cpu", type=float, default=0.3)
    sel.add_argument("--mem", type=float, default=0.5)
    sel.add_argument("--competency", type=float, default=0.0)
    sel.add_argument("--last-qoe", type=float, default=None)
    est = sub.add_parser("estimate-weights")
    est.add_argument("--group", choices=sorted(GROUP_KEYS))
    est.add_argument("--all", action="store_true")
    est.add_argument("--activate", action="store_true", help="use the result when METRIC_WEIGHTS_SOURCE=ewm")
    args = parser.parse_args()

    services = get_services()
    services.app.ensure_ready()  # creates any missing tables (including the adaptive-learning ones)
    store = services.experience_store()

    if args.cmd == "init":
        print(json.dumps(store.sync_index(), indent=2))
    elif args.cmd == "status":
        sync = store.sync_index()
        with store._db() as db:
            repo = AdaptiveLearningRepository(db)
            n = len(repo.list_transitions())
            active = repo.active_weight_sets()
        print(json.dumps({"transitions": n, "faiss_vectors": store.count(), "index": sync,
                          "weights_source": settings.metric_weights_source, "weights_version": store.weights().version,
                          "active_ewm_groups": {g: r["weights"] for g, r in active.items()}}, indent=2, default=str))
    elif args.cmd == "select":
        ctx = {"rtt_ms": args.rtt, "cpu": args.cpu, "mem": args.mem, "competency_num": args.competency,
               "last_qoe": args.last_qoe}
        candidates = [p["policy_id"] for p in services.app.list_policies()]
        default = (services.app.get_default_policy() or {}).get("policy_id", settings.default_policy_id)
        decision = PolicyEngine(store, random.Random(settings.policy_rng_seed)).select(ctx, candidates, default)
        print(json.dumps(decision.model_dump(), indent=2, default=str))
    else:
        groups = sorted(GROUP_KEYS) if args.all else [args.group]
        if groups == [None]:
            parser.error("give --group or --all")
        for group in groups:
            result, record = estimate_group(store, group, activate=args.activate)
            print(f"\n[{group}] {result.method} from {result.n_used}/{result.n_observations} observations"
                  + (f" — fallback: {result.fallback_reason}" if result.fallback_reason else "")
                  + (" — ACTIVE" if record["active"] else ""))
            for row in result.comparison():
                print(f"  {row['metric']:<4} baseline {row['baseline']:.3f}  derived {row['derived']:.3f}  "
                      f"({row['difference']:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
