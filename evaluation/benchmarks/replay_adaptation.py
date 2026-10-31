"""Results 3-4: edge vs cloud under network profiles, and adaptability under a changing network (trace replay).

    .venv\\Scripts\\python -m evaluation.benchmarks.replay_adaptation            # needs answer_matrix.csv
    .venv\\Scripts\\python -m evaluation.benchmarks.replay_adaptation --profiles good=college-wifi poor=phone-weak

Every outcome comes from the answer matrix (real answers, real timings, real resource use); the network model adds
what each emulated profile does to the cloud calls (network_model.py). Because every strategy is replayed on the same
questions, the same network draws and the same profile sequence, the comparison is paired and the oracle is exact.

A session is 5 questions of one topic under one network condition; the policy is chosen once per session, as in the
app. Session QoE/SRE/utility are computed with the app's own metric code (metrics.qoe/sre/utility, app weights):
AQ from retrieval score and answer accuracy, NQ from latency, network reliability, utilisation and availability,
SRE from CPU/memory, energy and bytes. EQ and UX need learners and are absent (renormalised; coverage reported).

Strategies: fixed P1-P5; an RTT rule (P3 when the probe is fast, else P1); random; the adaptive engine (the app's
PolicyEngine + ExperienceStore, ε-greedy, learning online from an empty memory, or warm-started on a training trace
built from a disjoint half of the questions); and the oracle (best policy per session in hindsight).

Outputs in evaluation/results/replay/: static_profiles.csv (Result 3), rtt_sweep.csv, sessions.csv and summary
tables (Result 4), stats.txt; figures via plot_results.py.
"""
import argparse
import random
import shutil
import sys
from dataclasses import dataclass
from typing import Any

from evaluation.benchmarks.isolation import RESULTS, disable_tracing, isolate

WORK = isolate("replay", fresh_copy=True, empty_experiences=True)

import pandas as pd  # noqa: E402

from agent.deps import get_services  # noqa: E402
from agent.experience_store import ExperienceStore  # noqa: E402
from agent.policies import POLICIES  # noqa: E402
from agent.policy_engine import PolicyEngine  # noqa: E402
from agent.session_metrics import qoe_bounds  # noqa: E402
from config import settings  # noqa: E402
from evaluation.benchmarks.network_model import (  # noqa: E402
    DEFAULT_PROFILES,
    NetworkProfile,
    cloud_call_extra_ms,
    cloud_fail_ms,
    probe_rtt_ms,
    probe_seen,
    profiles_from_csv,
)
from metrics.qoe import QoERaw, normalize_qoe  # noqa: E402
from metrics.reward import AdaptationAction, RuntimeState, compute_reward, score_state  # noqa: E402
from metrics.sre import SREInputs  # noqa: E402

OUT = RESULTS / "replay"
POLICY_IDS = ["P1", "P2", "P3", "P4", "P5"]
LOCAL_FALLBACK = {"P3": "P1", "P4": "P2", "P5": "P2"}  # what answers when the cloud call fails (same retrieval)
SESSION_Q = 5
COMPETENCY_NUM = settings.competency_levels.get("intermediate", 0.5)
TRACE = [("good", 10), ("poor", 10), ("blackhole", 4), ("moderate", 8), ("offline", 4), ("good", 8)]
ENGINES = ("v1", "v2")
RTT_RULE_MS = 1000.0  # probe = ~4 RTTs: below 1 s (RTT ≲ 250 ms) the rule uses the cloud


# ---- per-question outcome under a profile ----------------------------------------------------------------------

@dataclass
class Outcome:
    latency_ms: float
    correct: int
    retrieval_score: float
    attempts: int  # cloud calls attempted
    failures: int  # cloud calls that failed (outage)
    energy_wh: float
    net_bytes: float
    cpu: float
    mem: float
    provider: str


def _energy(row: Any, idle_gpu_w: float) -> float:
    gpu_above_idle = max(0.0, (row.energy_gpu_wh or 0.0) - idle_gpu_w * row.total_ms / 3.6e6)
    return (row.energy_cpu_wh_est or 0.0) + gpu_above_idle


def question_outcome(m: dict, qid: str, policy: str, p: NetworkProfile, base: NetworkProfile, rng: random.Random,
                     cloud_down: bool, idle_gpu_w: float) -> tuple[Outcome, bool]:
    """(outcome, cloud now known down). cloud_down: the router marked the cloud unhealthy after a failure (30 s)."""
    row = m[(qid, policy)]
    uses_cloud = row.n_cloud_calls > 0
    fail = cloud_fail_ms(p, settings.cloud_llm_timeout_s) if uses_cloud else None
    if uses_cloud and (fail is not None or cloud_down):
        local = m[(qid, LOCAL_FALLBACK[policy])]
        wait = 0.0 if cloud_down else fail  # skipped straight to the local model while marked unhealthy
        idle_wh = settings.energy_idle_power_w * wait / 3.6e6  # the device waits, drawing idle power
        return Outcome(local.total_ms + wait, int(local.correct or 0), local.retrieval_score, 0 if cloud_down else 1,
                       0 if cloud_down else 1, _energy(local, idle_gpu_w) + idle_wh, 0.0, local.cpu_mean,
                       local.mem_max, "local(fallback)"), True
    extra, net = 0.0, 0.0
    if uses_cloud:
        n = int(row.n_cloud_calls)
        for _ in range(n):
            extra += cloud_call_extra_ms(row.cloud_prompt_chars / n, row.cloud_answer_chars / n, p, base, rng)
        net = (row.cloud_prompt_chars + row.cloud_answer_chars) * 1.1 + 3000 * n
    return Outcome(row.total_ms + extra, int(row.correct or 0), row.retrieval_score, int(row.n_cloud_calls), 0,
                   _energy(row, idle_gpu_w), net, row.cpu_mean, row.mem_max, row.provider), cloud_down


def session_state(outcomes: list[Outcome], context: dict[str, float], probe_ok: bool) -> RuntimeState:
    n = len(outcomes)
    attempts = n + 1  # every answer + the context probe, as session_metrics counts network operations
    failed = sum(1 for o in outcomes if o.failures) + (0 if probe_ok else 1)
    raw = QoERaw(retrieval_score=sum(o.retrieval_score for o in outcomes) / n,
                 llm_quality=sum(o.correct for o in outcomes) / n,
                 latency_ms=sum(o.latency_ms for o in outcomes) / n,
                 network_reliability=(attempts - failed) / attempts,
                 utilization=max(max(o.cpu, o.mem) for o in outcomes),
                 availability=1.0)  # every answer came from an LLM (the local model answers in an outage)
    sre = SREInputs(cpu_usage=sum(o.cpu for o in outcomes) / n, cpu_max=settings.sre_cpu_max,
                    memory_usage=sum(o.mem for o in outcomes) / n, memory_max=settings.sre_memory_max,
                    energy_wh=sum(o.energy_wh for o in outcomes), energy_max_wh=settings.sre_energy_max_wh,
                    energy_estimated=True, bandwidth_bytes=sum(o.net_bytes for o in outcomes),
                    bandwidth_max_bytes=settings.sre_bandwidth_max_bytes, bandwidth_scope="app-level (cloud calls)")
    return RuntimeState(context=context, qoe_metrics=normalize_qoe(raw, qoe_bounds(settings)), sre_inputs=sre)


def run_session(m: dict, qids: list[str], policy: str, p: NetworkProfile, base: NetworkProfile, seed: int,
                idle_gpu_w: float, weights: Any) -> dict[str, Any]:
    rng = random.Random(seed)  # same draws for every policy of this session: paired comparison
    outcomes, down_until, clock = [], -1.0, 0.0
    for qid in qids:
        o, down = question_outcome(m, qid, policy, p, base, rng, clock < down_until, idle_gpu_w)
        if down and o.failures:
            down_until = clock + o.latency_ms + settings.llm_health_ttl_s * 1000
        clock += o.latency_ms + 15_000  # a learner reads the answer and asks the next question (~15 s)
        outcomes.append(o)
    context = {"rtt_ms": 0.0, "cpu": outcomes[0].cpu, "mem": outcomes[0].mem, "competency_num": COMPETENCY_NUM,
               "last_qoe": None}
    state = session_state(outcomes, context, probe_ok=p.outage is None)
    score = score_state(state, weights, settings.metric_missing_policy, baseline_qoe=settings.baseline_qoe,
                        baseline_sre=settings.baseline_sre, allow_baseline=False)
    lat = sorted(o.latency_ms for o in outcomes)
    metrics = {f"m_{k}": v for k, v in state.qoe_metrics.model_dump().items() if v is not None}
    groups = {f"g_{k}": v for k, v in (score.qoe_detail.components if score.qoe_detail else {}).items() if v is not None}
    sre_parts = {f"s_{k}": v for k, v in (score.sre_detail.components if score.sre_detail else {}).items()}
    return {"policy": policy, "qoe": score.qoe, "sre": score.sre, "utility": score.utility,
            **metrics, **groups, **sre_parts,
            "qoe_coverage": score.qoe_detail.coverage if score.qoe_detail else None,
            "latency_mean_ms": sum(lat) / len(lat), "latency_max_ms": lat[-1],
            "accuracy": sum(o.correct for o in outcomes) / len(outcomes),
            "cloud_failures": sum(o.failures for o in outcomes), "energy_wh": sum(o.energy_wh for o in outcomes),
            "net_bytes": sum(o.net_bytes for o in outcomes), "slow_answers": sum(o.latency_ms > 5000 for o in outcomes),
            "state": state}


# ---- strategies ------------------------------------------------------------------------------------------------

class AdaptiveEngine:
    """The app's PolicyEngine over a fresh ExperienceStore (its own DB rows and FAISS index)."""

    def __init__(self, tag: str, seed: int, version: str = "v1") -> None:
        services = get_services()
        self.app = services.app
        self.version = version
        self.cfg = settings.model_copy(update={"policy_engine": version})
        # what this engine's context records from the probe (v2 probes with a longer timeout)
        self.probe_timeout_s = self.cfg.policy_v2_probe_timeout_s if version == "v2" else self.cfg.probe_timeout_s
        index_dir = WORK / "indexes" / tag
        shutil.rmtree(index_dir, ignore_errors=True)
        self.store = ExperienceStore(self.app.session_factory, cfg=self.cfg, index_dir=index_dir)
        self.engine = PolicyEngine(self.store, random.Random(seed), self.cfg)
        self.student = self.app.create_learner(f"replay-{tag}")
        self.prev = RuntimeState(context={}, baseline=True)

    def choose(self, context: dict[str, Any]) -> tuple[str, Any]:
        decision = self.engine.select(context, POLICY_IDS, settings.default_policy_id)
        return decision.action_id, decision

    def learn(self, context: dict[str, Any], policy: str, s_next: RuntimeState, predicted: float | None) -> None:
        s_next = s_next.model_copy(update={"context": context})
        action = AdaptationAction(action_id=policy, config=POLICIES[policy].runtime_config())
        result = compute_reward(self.prev, s_next, action, self.store.weights(),
                                policy=settings.metric_missing_policy, baseline_qoe=settings.baseline_qoe,
                                baseline_sre=settings.baseline_sre)
        session = self.app.create_learning_session(self.student)
        self.store.record_transition(student_id=self.student, session_id=session, topic_id="replay", context=context,
                                     s=self.prev, s_next=s_next, action=action, result=result,
                                     predicted_utility=predicted)
        self.prev = s_next


def build_sessions(questions: pd.DataFrame, trace: list[tuple[str, int]], rng: random.Random) -> list[tuple[str, list[str]]]:
    by_topic = {t: list(g.qid) for t, g in questions.groupby("topic_id") if len(g) >= SESSION_Q}
    out = []
    for profile, count in trace:
        for _ in range(count):
            topic = rng.choice(sorted(by_topic))
            out.append((profile, rng.sample(by_topic[topic], SESSION_Q)))
    return out


def replay(m: dict, sessions: list[tuple[str, list[str]]], profiles: dict[str, NetworkProfile],
           base: NetworkProfile, rep: int, idle_gpu_w: float, engines: dict[str, AdaptiveEngine],
           weights: Any, label: str) -> list[dict[str, Any]]:
    """Every strategy over one session sequence; returns one row per (session, strategy)."""
    rows, rng_ctx, rng_random = [], random.Random(rep * 7919 + 1), random.Random(rep * 7919 + 2)
    last_qoe: dict[str, float | None] = {}
    for i, (pname, qids) in enumerate(sessions):
        p = profiles[pname]
        table = {pid: run_session(m, qids, pid, p, base, seed=rep * 100_003 + i * 101 + 7, idle_gpu_w=idle_gpu_w,
                                  weights=weights) for pid in POLICY_IDS}
        probe = probe_rtt_ms(p, rng_ctx, settings.rtt_fail_ms)  # raw; each engine applies its own timeout
        oracle = max(POLICY_IDS, key=lambda pid: table[pid]["utility"])
        choices = {**{pid: (pid, None) for pid in POLICY_IDS},
                   "rtt-rule": ("P3" if probe < RTT_RULE_MS else "P1", None),
                   "random": (rng_random.choice(POLICY_IDS), None), "oracle": (oracle, None)}
        contexts = {}
        for name, engine in engines.items():
            contexts[name] = {"rtt_ms": probe_seen(probe, engine.probe_timeout_s, settings.rtt_fail_ms),
                              "cpu": table["P1"]["state"].context["cpu"], "mem": table["P1"]["state"].context["mem"],
                              "competency_num": COMPETENCY_NUM, "last_qoe": last_qoe.get(name)}
            choices[name] = engine.choose(contexts[name])
        for strategy, (pid, decision) in choices.items():
            r = table[pid]
            rows.append({"run": label, "rep": rep, "session": i, "profile": pname, "strategy": strategy, "policy": pid,
                         **{k: v for k, v in r.items() if k not in ("state", "policy")},
                         "oracle_policy": oracle, "regret": table[oracle]["utility"] - r["utility"],
                         "probe_rtt_ms": probe,
                         "mode": decision.mode if decision else None,
                         "predicted": decision.predicted_value if decision else None})
        for name, engine in engines.items():
            pid, decision = choices[name]
            engine.learn(contexts[name], pid, table[pid]["state"], decision.predicted_value)
            last_qoe[name] = table[pid]["qoe"]
    return rows


# ---- Result 3: static profiles and an RTT sweep ------------------------------------------------------------------

def static_profiles(m: dict, questions: pd.DataFrame, profiles: dict[str, NetworkProfile], base: NetworkProfile,
                    idle_gpu_w: float, weights: Any, reps: int) -> pd.DataFrame:
    rows = []
    for rep in range(reps):
        sessions = build_sessions(questions, [(name, 30) for name in profiles], random.Random(10_000 + rep))
        for i, (pname, qids) in enumerate(sessions):
            for pid in POLICY_IDS:
                r = run_session(m, qids, pid, profiles[pname], base, seed=rep * 1_000_003 + i, idle_gpu_w=idle_gpu_w,
                                weights=weights)
                rows.append({"rep": rep, "session": i, "profile": pname, "policy": pid,
                             **{k: v for k, v in r.items() if k not in ("state", "policy")}})
    return pd.DataFrame(rows)


def rtt_sweep(m: dict, questions: pd.DataFrame, base: NetworkProfile, idle_gpu_w: float, weights: Any) -> pd.DataFrame:
    rows = []
    sessions = build_sessions(questions, [("x", 40)], random.Random(424242))
    for rtt in (20, 50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000):
        p = NetworkProfile(f"rtt{rtt}", rtt_ms=rtt, jitter_ms=rtt * 0.2, loss=0.0, down_mbps=10, up_mbps=4)
        for i, (_, qids) in enumerate(sessions):
            for pid in POLICY_IDS:
                r = run_session(m, qids, pid, p, base, seed=i * 31 + rtt, idle_gpu_w=idle_gpu_w, weights=weights)
                rows.append({"rtt_ms": rtt, "session": i, "policy": pid,
                             **{k: v for k, v in r.items() if k not in ("state", "policy")}})
    return pd.DataFrame(rows)


# ---- main --------------------------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", default=str(RESULTS / "answer_matrix.csv"))
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--profiles", nargs="*", default=[],
                        help="name=label pairs from network_profiles.csv, e.g. good=college-wifi poor=phone-weak")
    parser.add_argument("--base", default=None, help="label of the network the matrix was measured on")
    parser.add_argument("--skip-static", action="store_true", help="only the dynamic trace (Result 4)")
    parser.add_argument("--weights-file", default=None,
                        help="JSON {group: {key: weight}} overriding METRIC_WEIGHTS for scoring AND the engine")
    parser.add_argument("--out", default=None, help="output folder under evaluation/results (default replay)")
    args = parser.parse_args()
    disable_tracing()
    global OUT
    if args.out:
        OUT = RESULTS / args.out
    if args.weights_file:
        import json

        override = json.loads(open(args.weights_file, encoding="utf-8").read())
        settings.metric_weights = {**settings.metric_weights, **override}
        print(f"weights overridden from {args.weights_file}: {sorted(override)}")
    OUT.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.matrix)
    d = d[d.variant.isin(POLICY_IDS)]
    complete = d.groupby("qid").variant.nunique()
    d = d[d.qid.isin(complete[complete == len(POLICY_IDS)].index)]
    m = {(r.qid, r.variant): r for r in d.itertuples()}
    questions = d.drop_duplicates("qid")[["qid", "topic_id"]]
    print(f"{questions.qid.nunique()} questions with all five policies measured")

    profiles_csv = RESULTS / "network_profiles.csv"
    mapping = dict(x.split("=", 1) for x in args.profiles)
    profiles = profiles_from_csv(str(profiles_csv), mapping) if mapping else dict(DEFAULT_PROFILES)
    base = (profiles_from_csv(str(profiles_csv), {"base": args.base})["base"] if args.base
            else NetworkProfile("base", rtt_ms=30, down_mbps=50, up_mbps=20, source="assumed"))
    idle = pd.read_json(RESULTS / "idle_baseline.jsonl", lines=True)
    idle_gpu_w = float(idle.gpu_power_w.median()) if "gpu_power_w" in idle else 0.0
    weights = ExperienceStore(get_services().app.session_factory, index_dir=WORK / "indexes" / "_weights").weights()
    pd.DataFrame([{**p.__dict__} for p in [*profiles.values(), base]]).to_csv(OUT / "profiles_used.csv", index=False)

    if not args.skip_static:
        print("Result 3: static profiles ...")
        static_profiles(m, questions, dict(profiles), base, idle_gpu_w, weights, reps=5) \
            .to_csv(OUT / "static_profiles.csv", index=False)
        rtt_sweep(m, questions, base, idle_gpu_w, weights).to_csv(OUT / "rtt_sweep.csv", index=False)

    print("Result 4: dynamic trace ...")
    # warm start: train the engine on one trace over half the questions, test on traces over the other half
    rng = random.Random(7)
    by_topic = {t: sorted(g.qid) for t, g in questions.groupby("topic_id")}
    train_q, test_q = [], []
    for qids in by_topic.values():
        shuffled = rng.sample(qids, len(qids))
        train_q += shuffled[:len(qids) // 2]
        test_q += shuffled[len(qids) // 2:]
    train = questions[questions.qid.isin(train_q)]
    test = questions[questions.qid.isin(test_q)]
    rows = []
    for rep in range(args.reps):
        sessions = build_sessions(test, TRACE, random.Random(rep))
        # the engine as the app runs it (v1) and the improved one (v2), each from an empty memory ("cold") and
        # warm-started on a training trace over the other half of the questions ("warm")
        warm = {f"adaptive-{v}-warm": AdaptiveEngine(f"{v}-warm-{rep}", seed=rep, version=v) for v in ENGINES}
        replay(m, build_sessions(train, TRACE * 2, random.Random(10_000 + rep)), profiles, base, 10_000 + rep,
               idle_gpu_w, warm, weights, "training")
        cold = {f"adaptive-{v}-cold": AdaptiveEngine(f"{v}-cold-{rep}", seed=rep, version=v) for v in ENGINES}
        rows += replay(m, sessions, profiles, base, rep, idle_gpu_w, {**cold, **warm}, weights, "test")
        print(f"  rep {rep + 1}/{args.reps}", flush=True)
    sessions_df = pd.DataFrame(rows)
    sessions_df.to_csv(OUT / "sessions.csv", index=False)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
