"""Experience storage and retrieval for the adaptive loop: database records + FAISS similarity search.

- record_transition: persists (s, a, R, s_next) once per session. Person B's `experiences` row
  (via app.services.experience.store_experience), the `experience_transitions` details and the
  `experience_vectors` mapping are written in ONE database transaction; the FAISS vector is added
  after commit. If that add fails, the next sync() rebuilds the index from the database.
- neighbours: embeds the context, queries FAISS for the top-k states, and loads their full records.
- Weight resolution: config weights, overlaid with active EWM sets when METRIC_WEIGHTS_SOURCE=ewm.
"""
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy.orm import Session

from agent.policy_selection import STATE_KEYS, normalize_state
from app.db.repositories import AdaptiveLearningRepository
from app.services import experience as b_experience
from config import Settings, settings
from metrics.reward import AdaptationAction, RewardResult, RuntimeState
from metrics.weights import WeightSet
from services.experience_index import FaissExperienceIndex

logger = logging.getLogger(__name__)


class StateEmbedder:
    """Context → normalised [0, 1]^5 vector, identical for indexing and querying.

    The version string includes the normalisation parameters, so changing them re-embeds everything.
    """

    def __init__(self, *, rtt_max_ms: float, default_last_qoe: float) -> None:
        self.rtt_max_ms = rtt_max_ms
        self.default_last_qoe = default_last_qoe
        self.dim = len(STATE_KEYS)
        self.version = f"ctx5-rtt{int(rtt_max_ms)}-q{default_last_qoe:g}"

    def embed(self, context: dict[str, Any]) -> list[float]:
        vec = normalize_state(context, rtt_max_ms=self.rtt_max_ms, default_last_qoe=self.default_last_qoe)
        return [round(float(x), 6) for x in vec]


class ExperienceStore:
    """The adaptive loop's memory: full records in the app DB, state vectors in FAISS."""

    def __init__(self, session_factory: Callable[[], Session], *, cfg: Settings = settings,
                 index_dir: Any = "config") -> None:
        self.session_factory = session_factory
        self.cfg = cfg
        self.embedder = StateEmbedder(rtt_max_ms=cfg.rtt_max_ms, default_last_qoe=cfg.default_last_qoe)
        directory = cfg.experience_index_dir if index_dir == "config" else index_dir
        self.index = FaissExperienceIndex(self.embedder.dim, directory=directory, version=self.embedder.version)
        self._synced = False

    @contextmanager
    def _db(self) -> Iterator[Session]:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ---- index consistency ---------------------------------------------------------------------

    def sync_index(self) -> dict[str, Any]:
        """Make FAISS match the database: embed transitions that lack a vector for this version, then
        rebuild the index if its ids differ from the mapping table."""
        with self._db() as db:
            repo = AdaptiveLearningRepository(db)
            vectors = repo.list_vectors(self.embedder.version)
            known = {eid for _, eid, _ in vectors}
            transitions = repo.list_transitions()
            missing = [t for t in transitions if t["id"] not in known]
            if missing:  # new embedding version (or lost rows): re-embed every transition from its stored state
                repo.replace_vectors([(t["id"], self.embedder.embed(t["state"])) for t in transitions],
                                     self.embedder.version)
                vectors = repo.list_vectors(self.embedder.version)
        rebuilt = self.index.sync([(fid, vec) for fid, _, vec in vectors])
        self._synced = True
        return {"vectors": len(vectors), "rebuilt": rebuilt, "re_embedded": len(transitions) if missing else 0,
                "version": self.embedder.version}

    def _ensure_synced(self) -> None:
        if not self._synced:
            self.sync_index()

    # ---- weights ------------------------------------------------------------------------------

    def weights(self) -> WeightSet:
        """Config weights, with active EWM groups overlaid when METRIC_WEIGHTS_SOURCE=ewm."""
        base = WeightSet.from_config(self.cfg.metric_weights)
        if self.cfg.metric_weights_source != "ewm":
            return base
        with self._db() as db:
            active = AdaptiveLearningRepository(db).active_weight_sets()
        for group, record in active.items():
            if record["method"] == "ewm":
                base = base.replace(group, record["weights"], source="ewm")
        return base

    # ---- reads -------------------------------------------------------------------------------

    def neighbours(self, context: dict[str, Any], k: int) -> list[tuple[dict[str, Any], float]]:
        """Top-k most similar past states (by FAISS) with their full records and distances, closest first."""
        self._ensure_synced()
        hits = self.index.search(self.embedder.embed(context), k)
        with self._db() as db:
            repo = AdaptiveLearningRepository(db)
            id_map = repo.experience_ids_for([fid for fid, _ in hits])
            records = repo.records(list(id_map.values()))
        out = []
        for fid, distance in hits:
            record = records.get(id_map.get(fid, ""))
            if record is None:  # index drifted from the DB: resync next time
                self._synced = False
                continue
            out.append((record, distance))
        return out

    def count(self) -> int:
        self._ensure_synced()
        return self.index.ntotal

    def previous_state(self, student_id: str) -> tuple[RuntimeState, dict[str, Any] | None]:
        """The learner's last measured state (s for the next transition), or a baseline state."""
        with self._db() as db:
            last = AdaptiveLearningRepository(db).last_transition_for_student(student_id)
        if last is None:
            return RuntimeState(context={}, baseline=True), None
        return RuntimeState.model_validate(last["next_state"]), last

    # ---- writes ------------------------------------------------------------------------------

    def record_transition(self, *, student_id: str, session_id: str, topic_id: str, context: dict[str, Any],
                          s: RuntimeState, s_next: RuntimeState, action: AdaptationAction, result: RewardResult,
                          predicted_utility: float | None) -> tuple[dict[str, Any], bool]:
        """Persist (s, a, R, s_next) exactly once per session; returns (record, created)."""
        vector = self.embedder.embed(context)
        with self._db() as db:
            repo = AdaptiveLearningRepository(db)
            existing = repo.get_transition_by_session(session_id)
            if existing is not None:
                logger.info("transition for session %s already stored; not writing again", session_id)
                return existing, False
            evaluation = {"qoe_before": result.before.qoe, "qoe_after": result.after.qoe,
                          "sre_before": result.before.sre, "sre_after": result.after.sre, "reward": result.reward}
            experience = b_experience.store_experience(student_id, session_id, topic_id, action.action_id, context,
                                                       evaluation, db=db)
            repo.add_transition(
                experience_id=experience["id"], session_id=session_id, action=action.model_dump(mode="json"),
                prev_state=s.model_dump(mode="json"), next_state=s_next.model_dump(mode="json"),
                scores=result.model_dump(mode="json"), utility_before=result.before.utility,
                utility_after=result.after.utility, predicted_utility=predicted_utility,
                baseline_used=result.baseline_used, weights_version=result.weights_version)
            faiss_id = repo.add_vector(experience["id"], vector, self.embedder.version)
            record = repo.records([experience["id"]])[experience["id"]]
        try:
            self._ensure_synced()
            if faiss_id not in self.index.ids():
                self.index.add(faiss_id, vector)
        except Exception as exc:  # the DB has the record; the index is rebuilt from it on next use
            logger.error("FAISS add failed for %s, will resync: %s", experience["id"], exc)
            self._synced = False
        return record, True


def record_synthetic_transition(store: ExperienceStore, app: Any, *, policy_id: str, runtime_config: dict[str, Any],
                                context: dict[str, Any], qoe: float, sre: float, topic_id: str = "photosynthesis",
                                cfg: Settings = settings) -> dict[str, Any]:
    """Store a synthetic transition whose s_next has exactly the given QoE and SRE (tests, demos, seeding).

    It goes through the real path (new learner + session rows, compute_reward, record_transition), so
    the database, the FAISS mapping and the index stay consistent. Every QoE metric is set to `qoe`
    and every resource usage to (1 − sre) of its budget.
    """
    from metrics.qoe import QoEMetrics
    from metrics.reward import compute_reward
    from metrics.sre import SREInputs

    student = app.create_learner("synthetic")
    session = app.create_learning_session(student)
    s = RuntimeState(context={}, baseline=True)
    s_next = RuntimeState(
        context=context, session_id=session,
        qoe_metrics=QoEMetrics(**{k: qoe for k in QoEMetrics.model_fields}),
        sre_inputs=SREInputs(cpu_usage=1 - sre, memory_usage=1 - sre, energy_wh=(1 - sre) * 2.0, energy_max_wh=2.0,
                             bandwidth_bytes=(1 - sre) * 50e6, bandwidth_max_bytes=50e6))
    action = AdaptationAction(action_id=policy_id, config=runtime_config)
    result = compute_reward(s, s_next, action, store.weights(), policy=cfg.metric_missing_policy,
                            baseline_qoe=cfg.baseline_qoe, baseline_sre=cfg.baseline_sre)
    record, _ = store.record_transition(student_id=student, session_id=session, topic_id=topic_id, context=context,
                                        s=s, s_next=s_next, action=action, result=result, predicted_utility=None)
    return record
