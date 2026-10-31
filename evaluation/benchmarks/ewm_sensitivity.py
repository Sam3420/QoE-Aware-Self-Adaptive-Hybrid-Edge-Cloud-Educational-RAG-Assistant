"""Result 10: data-driven (Entropy Weight Method) metric weights, and how sensitive the conclusions are to weights.

    .venv\\Scripts\\python -m evaluation.benchmarks.ewm_sensitivity            # after replay_adaptation
    .venv\\Scripts\\python -m evaluation.benchmarks.replay_adaptation --skip-static \\
        --weights-file evaluation/results/ewm/ewm_weights.json --out replay_ewm  # the trace again under EWM weights
    .venv\\Scripts\\python -m evaluation.benchmarks.ewm_sensitivity --compare replay_ewm

1. EWM (metrics/ewm.py, the app's implementation) on the fixed-policy sessions of the dynamic trace, per group,
   over the metrics this benchmark measures (AQ: RQ, LQ; NQ: PL, NR, RU, AV; SRE: CPU, MEM, EN, NE; QoE top level:
   AQ, NQ; utility: QoE, SRE). Metrics that need learners (EQ, UX, SQ, TQ) keep their configured weights; the
   measured keys share the configured weight of their group, split by EWM (renormalisation makes this equivalent
   to EWM among the measured keys).
2. λ sweep: session utility re-weighted as λ·QoE + (1 − λ)·SRE for λ = 0.5 … 0.9 (decisions as made at λ = 0.7);
   the ranking of strategies and Kendall's τ against the configured λ = 0.7.
3. --compare <folder>: the full replay re-run under the EWM weights (scoring and the engine's own estimates).

Writes evaluation/results/ewm/{ewm_weights.json, ewm_weights.csv, lambda_sweep.csv, ewm_compare.csv}.
"""
import argparse
import json
import sys

from evaluation.benchmarks.isolation import RESULTS, isolate

isolate("ewm", fresh_copy=False)

import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

from config import settings  # noqa: E402
from metrics.ewm import entropy_weights  # noqa: E402

OUT = RESULTS / "ewm"
POLICIES = ["P1", "P2", "P3", "P4", "P5"]
GROUPS = {"aq": ("m_", ["RQ", "LQ"]), "nq": ("m_", ["PL", "NR", "RU", "AV"]), "sre": ("s_", ["CPU", "MEM", "EN", "NE"]),
          "qoe": ("g_", ["AQ", "NQ"])}


def derive(sessions: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    fixed = sessions[sessions.strategy.isin(POLICIES)]
    weights, rows = {}, []
    for group, (prefix, keys) in GROUPS.items():
        keys = [k for k in keys if f"{prefix}{k}" in fixed]
        obs = [{k: r[f"{prefix}{k}"] for k in keys} for _, r in fixed.iterrows()]
        base = settings.metric_weights[group]
        res = entropy_weights(group, obs, keys, {k: base[k] for k in keys})
        share = sum(base[k] for k in keys)
        weights[group] = {**base, **{k: round(res.weights[k] * share, 6) for k in keys}}
        for k in keys:
            rows.append({"group": group, "metric": k, "configured": base[k] / share, "ewm": res.weights[k],
                         "entropy": res.entropy[k], "method": res.method, "n": res.n_used})
    util = entropy_weights("utility", [{"QOE": q, "SRE": s} for q, s in zip(fixed.qoe, fixed.sre)], ["QOE", "SRE"],
                           settings.metric_weights["utility"])
    weights["utility"] = weights["reward"] = util.weights
    for k in ("QOE", "SRE"):
        rows.append({"group": "utility", "metric": k, "configured": settings.metric_weights["utility"][k],
                     "ewm": util.weights[k], "entropy": util.entropy[k], "method": util.method, "n": util.n_used})
    return weights, pd.DataFrame(rows)


def lambda_sweep(sessions: pd.DataFrame) -> pd.DataFrame:
    s = sessions[sessions.strategy != "oracle"]
    rows, ref = [], None
    for lam in (0.5, 0.6, 0.7, 0.8, 0.9):
        u = (lam * s.qoe + (1 - lam) * s.sre).groupby(s.strategy).mean().sort_values(ascending=False)
        ref = u if lam == 0.7 else ref
        rows.append({"lambda": lam, **{f"rank_{i + 1}": name for i, name in enumerate(u.index[:4])},
                     **{f"U_{k}": v for k, v in u.items()}})
    df = pd.DataFrame(rows)
    order = list(df[df["lambda"] == 0.7].filter(like="U_").T.squeeze().sort_values(ascending=False).index)
    df["kendall_tau_vs_0.7"] = [kendalltau([r[o] for o in order], list(range(len(order), 0, -1))).statistic
                                for _, r in df.iterrows()]
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", default=None, help="replay folder re-run under the EWM weights")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sessions = pd.read_csv(RESULTS / "replay" / "sessions.csv")
    weights, table = derive(sessions)
    (OUT / "ewm_weights.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")
    table.to_csv(OUT / "ewm_weights.csv", index=False)
    print(table.to_string(index=False))
    sweep = lambda_sweep(sessions)
    sweep.to_csv(OUT / "lambda_sweep.csv", index=False)
    print(sweep[["lambda", "rank_1", "rank_2", "rank_3", "kendall_tau_vs_0.7"]].to_string(index=False))
    if args.compare:
        other = pd.read_csv(RESULTS / args.compare / "sessions.csv")
        cmp = pd.DataFrame({"configured_weights": sessions.groupby("strategy").utility.mean(),
                            "ewm_weights": other.groupby("strategy").utility.mean(),
                            "configured_regret": sessions.groupby("strategy").regret.mean(),
                            "ewm_regret": other.groupby("strategy").regret.mean()}).reset_index()
        cmp["rank_configured"] = cmp.configured_weights.rank(ascending=False)
        cmp["rank_ewm"] = cmp.ewm_weights.rank(ascending=False)
        cmp.to_csv(OUT / "ewm_compare.csv", index=False)
        print(cmp.sort_values("rank_configured").to_string(index=False))
        print(f"Kendall τ between the two rankings: {kendalltau(cmp.rank_configured, cmp.rank_ewm).statistic:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
