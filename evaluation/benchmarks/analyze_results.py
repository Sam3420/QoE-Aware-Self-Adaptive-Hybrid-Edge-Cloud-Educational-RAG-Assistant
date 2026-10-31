"""Tables, statistics and figures for Results 1, 3, 4 and 5 (from answer_matrix.csv and replay/*.csv).

    .venv\\Scripts\\python -m evaluation.benchmarks.analyze_results

Writes evaluation/results/tables/*.csv + *.tex, evaluation/results/figures/*.pdf + *.png and
evaluation/results/stats.txt. Every number in the report should come from these files.
"""
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "evaluation" / "results"
TABLES, FIGURES = RESULTS / "tables", RESULTS / "figures"
POLICIES = ["P1", "P2", "P3", "P4", "P5"]
# categorical slots 1-5 of the validated palette (dataviz reference); markers/dashes are the secondary encoding
SERIES = {"slot": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
          "marker": ["o", "s", "^", "D", "v"], "dash": ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
PROFILE_ORDER = ["good", "moderate", "poor", "offline", "blackhole"]
ADAPTIVE = ["adaptive-v1-cold", "adaptive-v1-warm", "adaptive-v2-cold", "adaptive-v2-warm"]
_lines: list[str] = []


def say(text: str = "") -> None:
    _lines.append(text)
    print(text)


def style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def table(df: pd.DataFrame, name: str, caption: str, digits: int = 3) -> None:
    """CSV for the data, and a booktabs LaTeX table for the report."""
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=False)
    fmt = lambda v: (f"{v:.{digits}f}" if isinstance(v, float) and not math.isnan(v)  # noqa: E731
                     else "--" if isinstance(v, float) else str(v).replace("_", r"\_").replace("%", r"\%"))
    cols = " ".join("l" if df[c].dtype == object else "r" for c in df.columns)
    body = "\n".join(" & ".join(fmt(v) for v in row) + r" \\" for row in df.itertuples(index=False))
    head = " & ".join(str(c).replace("_", r"\_").replace("%", r"\%") for c in df.columns)
    (TABLES / f"{name}.tex").write_text(
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        f"\\caption{{{caption}}}\n\\label{{tab:{name}}}\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head} \\\\\n"
        f"\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n", encoding="utf-8")


def boot_ci(x: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    z, p = 1.96, k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def mcnemar(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    """Exact McNemar on paired correct/incorrect: (b-only correct, a-only correct, p)."""
    only_a, only_b = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n = only_a + only_b
    p = stats.binomtest(min(only_a, only_b), n, 0.5).pvalue if n else 1.0
    return only_b, only_a, p


# ---- Result 1 + 5: from the answer matrix ------------------------------------------------------------------------

def answer_matrix() -> None:
    path = RESULTS / "answer_matrix.csv"
    if not path.exists():
        say("answer_matrix.csv missing: run run_answer_matrix first")
        return
    d = pd.read_csv(path)
    d["correct"] = d["correct"].fillna(0)  # an answer that names no option counts as wrong
    n_q = d.qid.nunique()
    say(f"== Answer matrix: {n_q} questions, {len(d)} answers, variants {sorted(d.variant.unique())}")

    # Result 1: where the time goes, per policy (median and mean)
    rows = []
    for v in [*POLICIES, "RAG-local", "RAG-cloud", "LLM-local", "LLM-cloud"]:
        g = d[d.variant == v]
        if g.empty:
            continue
        rows.append({"variant": v, "n": len(g), "retrieval_ms": g.retrieve_ms.mean(),
                     "llm_ms": g.llm_ms.mean(), "check_ms": g.check_ms.fillna(0).mean(),
                     "total_ms_mean": g.total_ms.mean(), "total_ms_p50": g.total_ms.median(),
                     "total_ms_p95": g.total_ms.quantile(0.95), "cloud_calls": g.n_cloud_calls.mean(),
                     "tokens_out": g.completion_tokens.mean()})
    table(pd.DataFrame(rows), "latency_breakdown",
          "Answer latency by policy on the benchmark network (ms per question; retrieval, LLM and P5 check time).", 1)

    # Result 5: accuracy per variant with 95% Wilson intervals; CRAG ablation with McNemar
    acc = []
    for v in sorted(d.variant.unique(), key=lambda x: (x not in POLICIES, x)):
        g = d[d.variant == v]
        k, n = int(g.correct.sum()), len(g)
        lo, hi = wilson(k, n)
        acc.append({"variant": v, "n": n, "accuracy": k / n, "ci_low": lo, "ci_high": hi,
                    "parsed": g.parsed.mean(), "retrieval_score": g.retrieval_score.mean(),
                    "crag_retry_rate": (g.crag_retries > 0).mean(), "escalated": g.escalated.mean(),
                    "latency_ms": g.total_ms.mean()})
    table(pd.DataFrame(acc), "accuracy_by_variant",
          "Answer accuracy on the 150 multiple-choice benchmark items (95\\% Wilson interval).")
    wide = d.pivot_table(index="qid", columns="variant", values="correct")
    say("-- CRAG ablation (exact McNemar; b = only the second correct, c = only the first correct)")
    for a, b in [("RAG-local", "P2"), ("RAG-cloud", "P4"), ("LLM-local", "RAG-local"), ("LLM-cloud", "RAG-cloud"),
                 ("P1", "P3"), ("P2", "P4"), ("P2", "P5")]:
        if a in wide and b in wide:
            pair = wide[[a, b]].dropna()
            gain, loss, p = mcnemar(pair[a], pair[b])
            say(f"   {a} -> {b}: acc {pair[a].mean():.3f} -> {pair[b].mean():.3f}; "
                f"{b} better on {gain}, worse on {loss}; p = {p:.4f} (n = {len(pair)})")
    trig = d[d.variant.isin(["P2", "P4"])]
    say(f"   CRAG corrective retry triggered on {(trig.crag_retries > 0).mean():.1%} of P2/P4 answers; "
        f"retrieval score below threshold 0.65 before retry on {(d[d.variant == 'RAG-local'].retrieval_score < 0.65).mean():.1%}")
    p5 = d[d.variant == "P5"]
    if not p5.empty:
        say(f"   P5 escalated {p5.escalated.mean():.1%} of questions "
            f"(before answering {(p5.escalation_stage == 'before').mean():.1%}, after the check "
            f"{(p5.escalation_stage == 'after').mean():.1%}); mean check score {p5.check_score.mean():.3f}")

    # Energy and bytes per answer, above idle
    idle = pd.read_json(RESULTS / "idle_baseline.jsonl", lines=True).gpu_power_w.median()
    d["energy_wh"] = d.energy_cpu_wh_est.fillna(0) + (d.energy_gpu_wh.fillna(0) - idle * d.total_ms / 3.6e6).clip(lower=0)
    res = d[d.variant.isin(POLICIES)].groupby("variant").agg(
        energy_mwh=("energy_wh", lambda x: 1000 * x.mean()), gpu_w=("gpu_power_w", "mean"),
        cpu=("cpu_mean", "mean"), cloud_kb=("cloud_prompt_chars", lambda x: 0)).reset_index()
    res["cloud_kb"] = d[d.variant.isin(POLICIES)].groupby("variant").apply(
        lambda g: ((g.cloud_prompt_chars + g.cloud_answer_chars) * 1.1 / 1024).mean()).values
    table(res, "resources_by_policy",
          f"Resources per answer: energy above idle (mWh; CPU model estimate + NVML GPU above {idle:.1f} W idle), "
          "mean GPU board power, CPU utilisation and data sent to the cloud.", 2)

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    a = pd.DataFrame(acc)
    a = a[a.variant.isin(["LLM-local", "RAG-local", "P2", "LLM-cloud", "RAG-cloud", "P4"])]
    order = ["LLM-local", "RAG-local", "P2", "LLM-cloud", "RAG-cloud", "P4"]
    a = a.set_index("variant").loc[[o for o in order if o in a.variant.values or o in a.index]].reset_index()
    labels = {"LLM-local": "edge, no retrieval", "RAG-local": "edge, RAG", "P2": "edge, CRAG (P2)",
              "LLM-cloud": "cloud, no retrieval", "RAG-cloud": "cloud, RAG", "P4": "cloud, CRAG (P4)"}
    colors = [SERIES["slot"][0] if v.endswith("local") or v == "P2" else SERIES["slot"][1] for v in a.variant]
    x = np.arange(len(a))
    ax.bar(x, a.accuracy, width=0.55, color=colors, edgecolor="#fcfcfb", linewidth=2)
    ax.errorbar(x, a.accuracy, yerr=[a.accuracy - a.ci_low, a.ci_high - a.accuracy], fmt="none", ecolor=INK,
                elinewidth=1, capsize=3)
    for xi, v in zip(x, a.accuracy):
        ax.text(xi, v + 0.04, f"{v:.0%}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [labels[v] for v in a.variant], rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    style(ax, "Answer accuracy by retrieval mode (150 items, 95% CI)", "", "accuracy")
    save(fig, "accuracy_by_retrieval")


# ---- Result 3: static profiles and the RTT sweep --------------------------------------------------------------------

def static() -> None:
    path = RESULTS / "replay" / "static_profiles.csv"
    if not path.exists():
        return
    s = pd.read_csv(path)
    say("== Result 3: policies under each network profile (5 questions per session)")
    t = s.groupby(["profile", "policy"]).agg(latency_ms=("latency_mean_ms", "mean"),
                                             latency_max_ms=("latency_max_ms", "mean"),
                                             accuracy=("accuracy", "mean"), qoe=("qoe", "mean"), sre=("sre", "mean"),
                                             utility=("utility", "mean")).reset_index()
    t["profile"] = pd.Categorical(t.profile, [p for p in PROFILE_ORDER if p in set(t.profile)])
    t = t.sort_values(["profile", "policy"])
    table(t, "static_profiles", "Fixed policies under each network profile (session means).")
    best = t.loc[t.groupby("profile", observed=True).utility.idxmax()][["profile", "policy", "utility"]]
    say("   best fixed policy per profile: " + ", ".join(f"{r.profile}: {r.policy} ({r.utility:.3f})"
                                                       for r in best.itertuples()))

    sweep_path = RESULTS / "replay" / "rtt_sweep.csv"
    if not sweep_path.exists():
        return
    w = pd.read_csv(sweep_path).groupby(["rtt_ms", "policy"]).agg(
        latency=("latency_mean_ms", "mean"), utility=("utility", "mean")).reset_index()
    table(w, "rtt_sweep", "Mean answer latency and session utility as the round-trip time to the cloud grows.")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for i, pid in enumerate(["P1", "P3", "P5"]):
        g = w[w.policy == pid]
        for ax, col in zip(axes, ("latency", "utility")):
            ax.plot(g.rtt_ms, g[col] / (1000 if col == "latency" else 1), color=SERIES["slot"][i],
                    linestyle=SERIES["dash"][i], marker=SERIES["marker"][i], markersize=4, linewidth=2, label=pid)
            ax.annotate(pid, (g.rtt_ms.iloc[-1], g[col].iloc[-1] / (1000 if col == "latency" else 1)),
                        xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
    p1, p3 = (w[w.policy == p].set_index("rtt_ms").utility for p in ("P1", "P3"))
    cross = next((r for r in p1.index if p3[r] < p1[r]), None)
    if cross is not None:
        say(f"   crossover: the cloud policy P3 falls below the edge policy P1 in utility at RTT ≈ {cross} ms")
        for ax in axes:
            ax.axvline(cross, color=MUTED, linewidth=1, linestyle=":")
    style(axes[0], "Mean answer latency", "round-trip time to the cloud (ms)", "seconds")
    style(axes[1], "Session utility U", "round-trip time to the cloud (ms)", "U = 0.7·QoE + 0.3·SRE")
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, "rtt_sweep")


# ---- Result 4: adaptability under the dynamic trace ------------------------------------------------------------------

def dynamic() -> None:
    path = RESULTS / "replay" / "sessions.csv"
    if not path.exists():
        return
    s = pd.read_csv(path)
    say("== Result 4: dynamic network trace "
        f"({s.rep.nunique()} repetitions × {s.session.nunique()} sessions; profile sequence "
        f"{' → '.join(s[s.rep == 0].drop_duplicates('profile').profile)})")
    strategies = [*POLICIES, "rtt-rule", "random", *ADAPTIVE, "oracle"]
    rows = []
    for st in strategies:
        g = s[s.strategy == st]
        if g.empty:
            continue
        per_rep = g.groupby("rep").utility.mean().to_numpy()
        lo, hi = boot_ci(per_rep)
        rows.append({"strategy": st, "utility": per_rep.mean(), "ci_low": lo, "ci_high": hi,
                     "regret": g.regret.mean(), "qoe": g.qoe.mean(), "sre": g.sre.mean(),
                     "accuracy": g.accuracy.mean(), "latency_ms": g.latency_mean_ms.mean(),
                     "slow_answers_%": 100 * g.slow_answers.sum() / (5 * len(g)),
                     "cloud_failures": g.cloud_failures.mean(), "energy_mwh": 1000 * g.energy_wh.mean()})
    summary = pd.DataFrame(rows)
    table(summary, "dynamic_summary",
          "Strategies over the dynamic network trace (mean per session; 95\\% bootstrap CI over repetitions).")
    by_profile = s.pivot_table(index="strategy", columns="profile", values="utility", aggfunc="mean")
    by_profile = by_profile[[p for p in PROFILE_ORDER if p in by_profile.columns]].reindex(
        [x for x in strategies if x in by_profile.index]).reset_index()
    table(by_profile, "dynamic_by_profile", "Mean session utility of each strategy within each network profile.")

    # Friedman over all strategies except the oracle, blocks = (rep, session); Wilcoxon adaptive vs each, Holm
    blocks = s[s.strategy != "oracle"].pivot_table(index=["rep", "session"], columns="strategy", values="utility")
    blocks = blocks.dropna()
    fr = stats.friedmanchisquare(*[blocks[c] for c in blocks.columns])
    say(f"   Friedman over {len(blocks.columns)} strategies, {len(blocks)} blocks: χ² = {fr.statistic:.1f}, "
        f"p = {fr.pvalue:.2e}")
    for adaptive in ADAPTIVE:
        if adaptive not in blocks:
            continue
        per_rep = s[s.strategy != "oracle"].pivot_table(index="rep", columns="strategy", values="utility")
        pvals, effects = {}, {}
        for other in per_rep.columns:
            if other == adaptive:
                continue
            diff = per_rep[adaptive] - per_rep[other]
            pvals[other] = stats.wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
            effects[other] = diff.mean()
        adj = holm(pvals)
        say(f"   {adaptive} vs each (paired over repetitions, Wilcoxon, Holm-adjusted):")
        for other in sorted(pvals, key=lambda o: -effects[o]):
            say(f"      vs {other:13s} ΔU = {effects[other]:+.4f}  p_holm = {adj[other]:.4f}")

    # time to adapt: sessions after a profile change until the adaptive choice matches the segment's best fixed policy
    seg_best = s[s.strategy.isin(POLICIES)].groupby(["rep", "profile", "policy"]).utility.mean().reset_index()
    seg_best = seg_best.loc[seg_best.groupby(["rep", "profile"]).utility.idxmax()].set_index(["rep", "profile"]).policy
    for adaptive in ADAPTIVE:
        g = s[s.strategy == adaptive].sort_values(["rep", "session"])
        if g.empty:
            continue
        delays = []
        for rep, r in g.groupby("rep"):
            r = r.reset_index(drop=True)
            starts = r.index[r.profile != r.profile.shift()].tolist()
            for k, start in enumerate(starts[1:], 1):
                end = starts[k + 1] if k + 1 < len(starts) else len(r)
                target = seg_best[(rep, r.profile[start])]
                hit = next((i - start for i in range(start, end) if r.policy[i] == target), None)
                delays.append(end - start if hit is None else hit)
        if delays:
            say(f"   {adaptive}: sessions to adapt after a network change: median {np.median(delays):.1f}, "
                f"mean {np.mean(delays):.2f}, never within the segment {sum(1 for x in delays if x >= 4) / len(delays):.0%}"
                " (≥4 sessions)")
        share = g.groupby("profile").policy.value_counts(normalize=True).unstack(fill_value=0)
        table(share.reindex([p for p in PROFILE_ORDER if p in share.index]).reset_index(),
              f"policy_share_{adaptive.replace('-', '_')}", f"Policies chosen by {adaptive} in each network profile (share of sessions).")

    # prediction error of the engine (predicted vs observed utility)
    for adaptive in ADAPTIVE:
        g = s[(s.strategy == adaptive) & s.predicted.notna()]
        if not g.empty:
            say(f"   {adaptive}: mean |predicted − observed utility| = {(g.predicted - g.utility).abs().mean():.4f} "
                f"over {len(g)} greedy/explore decisions with an estimate")

    # timeline figure
    show = [x for x in ("oracle", "adaptive-v2-warm", "adaptive-v1-warm", "rtt-rule") if x in set(s.strategy)]
    best_fixed = summary[summary.strategy.isin(POLICIES)].sort_values("utility").strategy.iloc[-1]
    show.append(best_fixed)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    seq = s[(s.rep == 0) & (s.strategy == "oracle")].sort_values("session")[["session", "profile"]]
    shades = {"good": "#ffffff", "moderate": "#f3f2ee", "poor": "#e9e8e3", "offline": "#dcdbd5", "blackhole": "#d2d1ca"}
    for prof, grp in seq.groupby((seq.profile != seq.profile.shift()).cumsum()):
        x0, x1, name = grp.session.min() - 0.5, grp.session.max() + 0.5, grp.profile.iloc[0]
        ax.axvspan(x0, x1, color=shades.get(name, "#ffffff"), zorder=0)
        ax.text((x0 + x1) / 2, 1.02, name, transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=MUTED)
    for i, st in enumerate(show):
        g = s[s.strategy == st].groupby("session").utility.mean()
        label = f"{st} (best fixed)" if st == best_fixed else st
        ax.plot(g.index, g.values, color=SERIES["slot"][i], linestyle=SERIES["dash"][i], linewidth=2,
                marker=SERIES["marker"][i], markersize=3.5, label=label)
    style(ax, "Session utility over the dynamic trace (mean over repetitions)", "session", "utility U")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    save(fig, "dynamic_timeline")


def main() -> None:
    answer_matrix()
    static()
    dynamic()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "stats.txt").write_text("\n".join(_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    random.seed(0)
    main()
