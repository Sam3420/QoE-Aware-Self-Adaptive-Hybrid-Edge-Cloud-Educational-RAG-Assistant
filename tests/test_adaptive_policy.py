"""Policy engine, FAISS/DB experience store, EWM persistence and the full adaptive loop (synthetic, offline)."""
import random

import pytest

from agent.experience_store import ExperienceStore, record_synthetic_transition
from agent.policies import POLICIES
from agent.policy_engine import ActionEstimate, PolicyEngine, choose_action, estimate_action_values
from agent.weight_estimation import estimate_group
from app.db.repositories import AdaptiveLearningRepository
from metrics.reward import RuntimeState
from services.experience_index import FaissExperienceIndex
from tests.conftest import ANSWERS, CTX, pass_pretest

ADVANCED = {**CTX, "competency_num": 1.0}  # context of a session whose learner aces the pre-test
FAR = {"rtt_ms": 2000.0, "cpu": 0.95, "mem": 0.95, "competency_num": 1.0, "last_qoe": 0.0}


def seed(services, policy_id, value, context=CTX):
    return record_synthetic_transition(services.experience_store(), services.app, policy_id=policy_id,
                                       runtime_config=POLICIES[policy_id].runtime_config(), context=context,
                                       qoe=value, sre=value)


# ---- FAISS index ----------------------------------------------------------------------------------

def test_faiss_returns_most_similar_ids_and_persists(tmp_path):
    index = FaissExperienceIndex(2, directory=tmp_path, version="t")
    index.rebuild([(10, [0.0, 0.0]), (11, [1.0, 1.0]), (12, [0.1, 0.0])])
    hits = index.search([0.0, 0.05], 2)
    assert [i for i, _ in hits] == [10, 12] and hits[0][1] == pytest.approx(0.05, abs=1e-6)
    reloaded = FaissExperienceIndex(2, directory=tmp_path, version="t")
    assert reloaded.ids() == {10, 11, 12}
    assert reloaded.sync([(10, [0.0, 0.0])]) is True and reloaded.ids() == {10}  # DB is the source of truth
    with pytest.raises(ValueError):
        index.add(13, [0.0])


def test_store_maps_faiss_hits_to_full_db_records(services):
    near = seed(services, "P2", 0.9)
    seed(services, "P3", 0.4, context=FAR)
    neighbours = services.experience_store().neighbours(CTX, k=2)
    record, distance = neighbours[0]
    assert record["id"] == near["id"] and distance == pytest.approx(0.0, abs=1e-6)
    assert {"prev_state", "action", "reward", "next_state", "qoe_before", "qoe_after", "sre_before", "sre_after",
            "created_at"} <= set(record)
    assert record["action"]["action_id"] == "P2" and record["next_state"]["qoe_metrics"]["PT"] == 0.9


def test_transition_is_persisted_exactly_once(services):
    store = services.experience_store()
    record = seed(services, "P1", 0.6)
    s = RuntimeState.model_validate(record["prev_state"])
    s_next = RuntimeState.model_validate(record["next_state"])
    from metrics.reward import AdaptationAction, RewardResult

    again, created = store.record_transition(
        student_id=record["student_id"], session_id=record["session_id"], topic_id="photosynthesis", context=CTX,
        s=s, s_next=s_next, action=AdaptationAction(**record["action"]),
        result=RewardResult.model_validate(record["scores"]), predicted_utility=None)
    assert not created and again["id"] == record["id"]
    assert store.count() == 1 and len(services.app.get_experiences()) == 1


def test_index_is_rebuilt_from_the_database(services):
    seed(services, "P1", 0.6)
    seed(services, "P2", 0.7)
    fresh = ExperienceStore(services.app.session_factory, index_dir=None)  # empty in-memory index
    assert fresh.count() == 2  # sync on first use rebuilt it from experience_vectors


# ---- estimation and selection -------------------------------------------------------------------------

def rec(policy, eid="e"):
    return {"id": eid, "policy_id": policy, "session_id": eid}


def test_estimates_use_only_each_actions_own_neighbours():
    neighbours = [(rec("P1", "a"), 0.0), (rec("P2", "b"), 0.1), (rec("P2", "c"), 0.5)]
    values = {"a": 0.2, "b": 0.9, "c": 0.5}
    est = estimate_action_values(neighbours, ["P1", "P2", "P3"], value_fn=lambda r: values[r["id"]], sigma=0.25,
                                 min_support=1)
    assert est["P1"].estimated_value == 0.2  # the nearest neighbour's value is not copied to other actions
    assert est["P2"].support == 2 and 0.5 < est["P2"].estimated_value < 0.9  # similarity-weighted toward b
    assert est["P2"].estimated_value > (0.9 + 0.5) / 2
    assert est["P3"].estimated_value is None and est["P3"].support == 0
    strict = estimate_action_values(neighbours, ["P1"], value_fn=lambda r: values[r["id"]], sigma=0.25, min_support=2)
    assert strict["P1"].estimated_value is None
    far = estimate_action_values([(rec("P1", "a"), 1.5)], ["P1"], value_fn=lambda r: 1.0, sigma=0.25, min_support=1,
                                 min_weight_sum=0.1)
    assert far["P1"].estimated_value is None and far["P1"].support == 1  # weight ≈ 1.5e-8: not trusted


def _est(**values):
    return {a: ActionEstimate(action_id=a, estimated_value=v, support=0 if v is None else 1, weight_sum=1.0)
            for a, v in values.items()}


def test_choose_action_rules():
    rng = random.Random(0)  # first draw 0.844
    args = dict(n_history=10, min_experiences=3, epsilon=0.2, target="utility")
    assert choose_action(_est(P1=0.3, P2=0.8, P3=None), ["P1", "P2", "P3"], "P1", rng=rng, **args)[:2] == ("P2", "greedy")
    assert choose_action(_est(P1=0.5, P2=0.5), ["P1", "P2"], "P1", rng=random.Random(0), **args)[0] == "P1"  # tie
    assert choose_action(_est(P1=None, P2=None), ["P1", "P2"], "P1", rng=random.Random(0), **args)[1] == "fallback"
    cold = choose_action(_est(P2=0.9), ["P1", "P2"], "P1", rng=random.Random(0), **{**args, "n_history": 2})
    assert cold[:2] == ("P1", "cold_start")
    explore = [choose_action(_est(P1=0.3, P2=0.8, P3=None), ["P1", "P2", "P3"], "P1", rng=random.Random(s),
                             **{**args, "epsilon": 1.0}) for s in range(6)]
    assert all(m == "exploration" and a != "P2" for a, m, _ in explore) and {a for a, _, _ in explore} >= {"P3"}
    seq = [choose_action(_est(P1=0.3, P2=0.8), ["P1", "P2"], "P1", rng=random.Random(7), **{**args, "epsilon": 0.5})[0]
           for _ in range(2)]
    assert seq[0] == seq[1]  # seeded → reproducible


def test_engine_selects_highest_estimated_utility(services):
    for policy, value in [("P1", 0.3), ("P2", 0.9), ("P2", 0.8), ("P4", 0.6)]:
        seed(services, policy, value)
    seed(services, "P3", 1.0, context=FAR)  # excellent but in a very different state
    decision = PolicyEngine(services.experience_store(), random.Random(0)).select(CTX, ["P1", "P2", "P3", "P4"], "P1")
    assert decision.action_id == "P2" and decision.mode == "greedy"
    assert decision.estimates["P2"].estimated_value == pytest.approx(0.85, abs=1e-6)
    p3 = decision.estimates["P3"]
    assert p3.support == 1 and p3.weight_sum < 0.1 and p3.estimated_value is None  # far evidence is not trusted
    assert decision.predicted_value == decision.estimates["P2"].estimated_value


def test_empty_history_cold_start(services):
    decision = PolicyEngine(services.experience_store(), random.Random(0)).select(CTX, ["P1", "P2"], "P1")
    assert decision.action_id == "P1" and decision.mode == "cold_start" and decision.neighbours == []


# ---- full loop ------------------------------------------------------------------------------------

def _session(runner, learner, questions=("How do plants make glucose from light?",), rating=4):
    out = runner.start_session(learner, "photosynthesis")
    tid = out["thread_id"]
    if out["interrupt"]["type"] == "pre_test":
        pass_pretest(runner, tid, ANSWERS)
    for q in questions:
        runner.resume(tid, {"query": q})
    runner.resume(tid, {"end": True})
    return runner.resume(tid, {"answers": ANSWERS, "rating": rating, "comment": "clear and helpful"})["final_state"]


def test_full_loop_second_session_uses_first_as_previous_state(runner, learner, services):
    first = _session(runner, learner)
    ev1 = first["evaluation"]
    assert ev1["status"] == "completed" and ev1["baseline_used"]
    second = _session(runner, learner, questions=("What does chlorophyll do?", "Why are leaves green?"), rating=2)
    ev2 = second["evaluation"]
    assert not ev2["baseline_used"]
    assert ev2["qoe_before"] == pytest.approx(ev1["qoe_after"]) and ev2["sre_before"] == pytest.approx(ev1["sre_after"])
    assert ev2["reward"] == pytest.approx(0.7 * ev2["delta_qoe"] + 0.3 * ev2["delta_sre"], abs=1e-6)
    assert ev2["metrics"]["SR"] == 0.25 and ev2["metrics"]["FS"] == 1.0  # rating 2/5, positive comment
    store = services.experience_store()
    assert store.count() == 2 == len(services.app.get_experiences())
    with store._db() as db:
        vectors = AdaptiveLearningRepository(db).list_vectors(store.embedder.version)
    assert {fid for fid, _, _ in vectors} == store.index.ids()  # FAISS consistent with the DB


def test_unexecuted_session_is_not_stored(runner, learner, services):
    final = _session(runner, learner, questions=())
    assert final["evaluation"]["status"] == "not_executed"
    assert any("experience not stored" in e for e in final["session_errors"])
    assert services.app.get_experiences() == [] and services.experience_store().count() == 0


def test_prediction_and_observation_are_stored_separately(runner, learner, services):
    for value in (0.9, 0.8, 0.85):
        seed(services, "P1", value, context=ADVANCED)
    final = _session(runner, learner)
    ev = final["evaluation"]
    assert ev["predicted_utility"] == pytest.approx(0.85, abs=1e-6)  # estimated before execution
    assert ev["utility_after"] != ev["predicted_utility"]  # measured after execution
    with services.experience_store()._db() as db:
        stored = AdaptiveLearningRepository(db).get_transition_by_session(final["session_id"])
    assert stored["predicted_utility"] == ev["predicted_utility"] and stored["utility_after"] == ev["utility_after"]


# ---- EWM persistence ------------------------------------------------------------------------------

def test_ewm_estimation_is_persisted_and_only_active_when_enabled(services, monkeypatch):
    store = services.experience_store()
    rng = random.Random(1)
    for _ in range(12):
        seed(services, rng.choice(["P1", "P2"]), round(rng.uniform(0.2, 0.9), 2))
    result, record = estimate_group(store, "utility", activate=True)
    # QoE = SRE = value in every synthetic transition, so both columns vary identically → equal weights
    assert result.method == "ewm" and result.weights == pytest.approx({"QOE": 0.5, "SRE": 0.5})
    assert record["active"] and record["n_used"] == 12 and record["window_start"] is not None
    assert store.weights().get("utility") == {"QOE": 0.7, "SRE": 0.3}  # source=fixed: not applied
    monkeypatch.setattr(store.cfg, "metric_weights_source", "ewm")
    assert store.weights().get("utility") == pytest.approx({"QOE": 0.5, "SRE": 0.5})
    monkeypatch.setattr(store.cfg, "ewm_min_observations", 100)
    few, rec_few = estimate_group(store, "utility", activate=True)
    assert few.method == "fallback" and not rec_few["active"]  # a fallback is stored but never activated
    assert store.weights().get("utility") == pytest.approx({"QOE": 0.5, "SRE": 0.5})  # previous EWM set stays


# ---- observability --------------------------------------------------------------------------------

def test_tracing_is_noop_without_keys_and_never_breaks_the_loop(monkeypatch, runner, learner):
    import agent.observability as obs

    monkeypatch.setattr(obs, "_client_checked", False)
    monkeypatch.setattr(obs, "_client", None)
    with obs.observe("x") as span:  # no LANGFUSE keys configured
        span.update(output=1)
        span.score(name="s", value=1.0)
    assert obs._client is None

    calls = []

    class FakeSpan:
        def update(self, **kw):
            calls.append(("update", kw))

        def score(self, **kw):
            calls.append(("score", kw))

    class FakeCM:
        def __enter__(self):
            return FakeSpan()

        def __exit__(self, *a):
            calls.append(("end", None))

    class FakeClient:
        def start_as_current_observation(self, **kw):
            calls.append(("start", kw["name"]))
            return FakeCM()

    monkeypatch.setattr(obs, "_client_checked", True)
    monkeypatch.setattr(obs, "_client", FakeClient())
    final = _session(runner, learner)
    names = [c[1] for c in calls if c[0] == "start"]
    assert names == ["select_policy", "evaluate_session", "store_experience"]
    assert {c[1]["name"] for c in calls if c[0] == "score"} == {"qoe", "sre", "reward"}

    class BrokenClient:
        def start_as_current_observation(self, **kw):
            raise RuntimeError("langfuse down")

    monkeypatch.setattr(obs, "_client", BrokenClient())
    assert _session(runner, learner)["evaluation"]["status"] == "completed" and final["reward"] is not None
