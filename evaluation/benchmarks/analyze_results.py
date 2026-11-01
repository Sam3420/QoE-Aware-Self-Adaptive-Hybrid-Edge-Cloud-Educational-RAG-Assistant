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
# Report names (internal ids stay as above): QAPS = QoE-Aware Policy Selector, the proposed selector (POLICY_ENGINE=v2);
# the baseline is the original experience-based kNN selector (v1). "cold" = empty memory, "warm" = pre-trained.
LABELS = {"adaptive-v1-cold": "Baseline kNN selector (cold)", "adaptive-v1-warm": "Baseline kNN selector (warm)",
          "adaptive-v2-cold": "QAPS, proposed (cold)", "adaptive-v2-warm": "QAPS, proposed (warm)",
          "rtt-rule": "RTT threshold rule", "random": "Random", "oracle": "Oracle (hindsight)",
          "P1": "P1 edge-light", "P2": "P2 edge-thorough", "P3": "P3 cloud-light", "P4": "P4 cloud-thorough",
          "P5": "P5 edge-escalate"}


def label(name: str) -> str:
    return LABELS.get(name, name)
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
    if "strategy" in df:  # report names for the strategies (the CSVs keep them too, for consistency)
        df = df.assign(strategy=df["strategy"].map(label))
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
    ax.errorbar(x, a.accuracy, yerr=[(a.accuracy - a.ci_low).clip(lower=0), (a.ci_high - a.accuracy).clip(lower=0)],
                fmt="none", ecolor=INK,
                elinewidth=1, capsize=3)
    for xi, v in zip(x, a.accuracy):
        ax.text(xi, v + 0.04, f"{v:.0%}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [labels[v] for v in a.variant], rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    style(ax, f"Answer accuracy by retrieval mode (multiple-choice items, n = {int(a.n.min())}–{int(a.n.max())}, 95% CI)",
          "", "accuracy")
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
        say(f"   {label(adaptive)} vs each (paired over repetitions, Wilcoxon, Holm-adjusted):")
        for other in sorted(pvals, key=lambda o: -effects[o]):
            say(f"      vs {label(other):30s} ΔU = {effects[other]:+.4f}  p_holm = {adj[other]:.4f}")

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
            say(f"   {label(adaptive)}: sessions to adapt after a network change: median {np.median(delays):.1f}, "
                f"mean {np.mean(delays):.2f}, never within the segment {sum(1 for x in delays if x >= 4) / len(delays):.0%}"
                " (≥4 sessions)")
        share = g.groupby("profile").policy.value_counts(normalize=True).unstack(fill_value=0)
        table(share.reindex([p for p in PROFILE_ORDER if p in share.index]).reset_index(),
              f"policy_share_{adaptive.replace('-', '_')}", f"Policies chosen by {adaptive} in each network profile (share of sessions).")

    # prediction error of the engine (predicted vs observed utility)
    for adaptive in ADAPTIVE:
        g = s[(s.strategy == adaptive) & s.predicted.notna()]
        if not g.empty:
            say(f"   {label(adaptive)}: mean |predicted − observed utility| = {(g.predicted - g.utility).abs().mean():.4f} "
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
        ax.text((x0 + x1) / 2, 0.97, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8,
                color=MUTED)
    for i, st in enumerate(show):
        g = s[s.strategy == st].groupby("session").utility.mean()
        name = f"{label(st)} (best fixed)" if st == best_fixed else label(st)
        ax.plot(g.index, g.values, color=SERIES["slot"][i], linestyle=SERIES["dash"][i], linewidth=2,
                marker=SERIES["marker"][i], markersize=3.5, label=name)
    style(ax, "Session utility over the dynamic trace (mean over repetitions)", "session", "utility U")
    ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=11, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    save(fig, "dynamic_timeline")


# ---- Result 5b: learner doubts (faithfulness, declining off-topic) ------------------------------------------------

def doubts() -> None:
    path = RESULTS / "doubts_matrix.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    say(f"== Result 5b: learner doubts ({d[d.kind == 'doubt'].qid.nunique()} in scope, "
        f"{d[d.kind == 'off_topic'].qid.nunique()} off-topic), variants {sorted(d.variant.unique())}")
    rows = []
    for (v, kind), g in d.groupby(["variant", "kind"]):
        rows.append({"variant": v, "set": kind, "n": len(g), "retrieval_score": g.retrieval_score.mean(),
                     "below_threshold_%": 100 * (g.retrieval_score < 0.65).mean(),
                     "crag_retry_%": 100 * (g.crag_retries > 0).mean(), "declined_%": 100 * g.declined.mean(),
                     "faithfulness": g.faithfulness.mean(), "judged": int(g.faithfulness.notna().sum()),
                     "latency_ms": g.total_ms.mean()})
    table(pd.DataFrame(rows), "doubts_summary",
          "Learner doubts and off-topic questions: retrieval, CRAG retries, declines and judged faithfulness.", 2)
    wide = d[d.kind == "doubt"].pivot_table(index="qid", columns="variant", values="faithfulness")
    for a, b in [("RAG-local", "P2"), ("RAG-cloud", "P4")]:
        if a in wide and b in wide:
            pair = wide[[a, b]].dropna()
            diff = pair[b] - pair[a]
            p = stats.wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
            say(f"   faithfulness {a} {pair[a].mean():.3f} -> {b} {pair[b].mean():.3f} "
                f"(Wilcoxon p = {p:.4f}, n = {len(pair)})")
    off = d[d.kind == "off_topic"]
    if not off.empty:
        say("   off-topic correctly declined: " + ", ".join(f"{v} {g.declined.mean():.0%}"
                                                          for v, g in off.groupby("variant")))


# ---- Result 9b: cost of answer scaffolding ------------------------------------------------------------------------

def scaffold() -> None:
    path = RESULTS / "coach" / "scaffold.csv"
    if not path.exists():
        return
    s = pd.read_csv(path)
    t = s.groupby(["variant", "scaffold"]).agg(n=("qid", "size"), latency_ms=("total_ms", "mean"),
                                              tokens_out=("completion_tokens", "mean"),
                                              answer_chars=("answer_chars", "mean")).reset_index()
    for v, g in t.groupby("variant"):
        base = g[g.scaffold == "none"].latency_ms.iloc[0] if (g.scaffold == "none").any() else None
        if base:
            t.loc[g.index, "latency_vs_plain_%"] = 100 * (g.latency_ms / base - 1)
    table(t, "scaffold_cost", "Cost of the coach's answer scaffolding (30 learner doubts per row).", 1)
    say("== Result 9b: scaffolding cost: " + "; ".join(
        f"{r.variant} {r.scaffold}: {r.latency_ms:.0f} ms, {r.tokens_out:.0f} tokens" for r in t.itertuples()))


# ---- extra figures for the report -----------------------------------------------------------------------------------

HIGHLIGHT, OTHER = SERIES["slot"][0], "#b9b8b1"


def fig_strategies() -> None:
    """Headline: mean session utility per strategy with 95% CI; the proposed selector highlighted."""
    path = TABLES / "dynamic_summary.csv"
    if not path.exists():
        return
    d = pd.read_csv(path).sort_values("utility")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = [HIGHLIGHT if "QAPS" in st else (SERIES["slot"][1] if "Baseline" in st else OTHER) for st in d.strategy]
    y = np.arange(len(d))
    ax.barh(y, d.utility, height=0.6, color=colors, edgecolor="#fcfcfb", linewidth=2)
    ax.errorbar(d.utility, y, xerr=[(d.utility - d.ci_low).clip(lower=0), (d.ci_high - d.utility).clip(lower=0)],
                fmt="none", ecolor=INK, elinewidth=1, capsize=2)
    for yi, v, hi in zip(y, d.utility, d.ci_high):
        ax.text(hi + 0.002, yi, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, d.strategy)
    lo = d.utility.min()
    ax.set_xlim(max(0.0, lo - 0.05), d.ci_high.max() + 0.02)
    style(ax, "Mean session utility over the dynamic trace (95% CI)", "utility U", "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    save(fig, "strategy_utility")


def fig_latency() -> None:
    """Where an answer's time goes, per policy (median of the answer matrix)."""
    path = RESULTS / "answer_matrix.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    d = d[d.variant.isin(POLICIES)]
    g = d.groupby("variant").agg(retrieval=("retrieve_ms", "median"), check=("check_ms", lambda x: x.fillna(0).median()),
                                 total=("total_ms", "median"), llm=("llm_ms", "median")).reindex(POLICIES)
    g["other"] = (g.total - g.retrieval - g.llm - g.check).clip(lower=0)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(g))
    left = np.zeros(len(g))
    for i, (col, name) in enumerate([("retrieval", "retrieval"), ("llm", "answer generation"),
                                     ("check", "grounding check (P5)"), ("other", "other")]):
        ax.barh(y, g[col] / 1000, left=left, height=0.6, color=SERIES["slot"][i], edgecolor="#fcfcfb", linewidth=2,
                label=name)
        left += g[col].to_numpy() / 1000
    for yi, t in zip(y, g.total):
        ax.text(t / 1000 + 0.05, yi, f"{t / 1000:.2f} s", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [label(p) for p in g.index])
    ax.invert_yaxis()
    style(ax, "Where the time of one answer goes (median, benchmark network)", "seconds", "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    save(fig, "latency_breakdown")


def fig_doubts() -> None:
    """Faithfulness on learner doubts, and declining off-topic questions, per retrieval mode."""
    path = RESULTS / "doubts_matrix.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    order = [v for v in ("RAG-local", "P2", "RAG-cloud", "P4") if v in set(d.variant)]
    names = {"RAG-local": "edge, RAG", "P2": "edge, CRAG", "RAG-cloud": "cloud, RAG", "P4": "cloud, CRAG"}
    faith = d[d.kind == "doubt"].groupby("variant").faithfulness.mean().reindex(order)
    declined = d[d.kind == "off_topic"].groupby("variant").declined.mean().reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    colors = [SERIES["slot"][0] if v in ("RAG-local", "P2") else SERIES["slot"][1] for v in order]
    for ax, values, title, fmt in ((axes[0], faith, "Faithfulness to the lesson (30 doubts)", "{:.2f}"),
                                   (axes[1], declined, "Off-topic questions declined (10)", "{:.0%}")):
        x = np.arange(len(order))
        ax.bar(x, values.fillna(0), width=0.55, color=colors, edgecolor="#fcfcfb", linewidth=2)
        for xi, v in zip(x, values):
            ax.text(xi, (0 if pd.isna(v) else v) + 0.03, "–" if pd.isna(v) else fmt.format(v), ha="center", fontsize=8,
                    color=INK)
        ax.set_xticks(x, [names[v] for v in order])
        ax.set_ylim(0, 1.12)
        style(ax, title, "", "")
    save(fig, "doubts_faithfulness")


def speech() -> None:
    path = RESULTS / "speech" / "roundtrip.csv"
    if not path.exists():
        return
    s = pd.read_csv(path)
    t = s.groupby(["backend", "language"]).agg(n=("wer", "size"), wer=("wer", "mean"), stt_ms=("stt_ms", "median"),
                                               tts_ms=("tts_ms", "median"), audio_s=("audio_s", "mean")).reset_index()
    table(t, "speech_summary", "Speech round trip (Kokoro TTS → STT): word error rate and median latency.", 3)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    names = [f"{b} ({'English' if l == 'en' else 'Hindi'})" for b, l in zip(t.backend, t.language)]
    x = np.arange(len(t))
    ax.bar(x, t.wer * 100, width=0.5, color=[SERIES["slot"][0] if b == "sravaani" else SERIES["slot"][1]
                                              for b in t.backend], edgecolor="#fcfcfb", linewidth=2)
    for xi, w, ms in zip(x, t.wer, t.stt_ms):
        ax.text(xi, w * 100 + 0.4, f"{w:.1%} · {ms:.0f} ms", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, names)
    ax.set_ylim(0, (t.wer.max() * 100) * 1.2)
    style(ax, "Speech-to-text word error rate (synthetic voices)", "", "WER (%)")
    save(fig, "speech_wer")


def main() -> None:
    answer_matrix()
    static()
    dynamic()
    doubts()
    scaffold()
    speech()
    fig_strategies()
    fig_latency()
    fig_doubts()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "stats.txt").write_text("\n".join(_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    random.seed(0)
    main()
