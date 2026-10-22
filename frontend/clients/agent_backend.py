"""In-process access to the LangGraph learning-session agent and the real services it is wired with.

The agent has no HTTP API: docs/agent_runner.md names agent/runner.py as the UI entry point. This
module drives that runner with the same real services as scripts/run_session.py
(agent.deps.get_services → make_real_services: FAISS + CRAG, Ollama/HF/extractive router, psutil
monitor, app DB). It never installs the fake services.
"""
import json
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.deps import Services, get_services
from agent.graph import build_graph, get_checkpointer
from agent.runner import SessionNotFound, SessionRunner
from app.core.config import get_settings as get_app_settings
from config import settings
from knowledge_prep.catalog import catalog_order
from services.crag import RetrievalResult
from services.monitoring import ProbeError
from services.vector_store import IndexMeta

ROUTES = ("local", "cloud")


@dataclass(frozen=True)
class ProviderStatus:
    """Availability of one LLM route, as the agent's own router reports it."""

    provider: str
    model: str
    available: bool
    how_checked: str


@dataclass(frozen=True)
class IndexStatus:
    """Whether a topic has a FAISS index the configured embedder can query."""

    topic: str
    exists: bool
    source: Path | None
    embedding_model: str | None = None
    n_chunks: int | None = None
    dim: int | None = None
    compatible: bool = False
    error: str | None = None


@dataclass
class StepResult:
    """Outcome of one runner call: the runner's return value, the nodes it ran, and any exception."""

    out: dict[str, Any] | None
    trace: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0
    error: str | None = None
    error_traceback: str | None = None


def describe_node(node: str, update: dict[str, Any]) -> str:
    """One-line summary of a node's real state update, for the execution trace."""
    if node == "load_profile" and "profile" in update:
        return f"learner {update['profile'].get('display_name')!r}, language {update['profile'].get('language')}"
    if node == "plan_pretest":
        plan = update.get("pretest_plan") or {}
        if plan.get("mode") == "unavailable":
            return f"no pre-test: {plan.get('reason')}"
        return (f"{plan.get('mode')} learner: {len(update.get('test_items', []))} questions"
                + (f", weak concepts {', '.join(plan['weak_concepts'])}" if plan.get("weak_concepts") else ""))
    if node == "pre_test":
        if "pre_score" in update:
            return f"pre_score={update['pre_score']:.2f} → competency={update['competency']}"
        return "; ".join(update.get("session_errors", [])) or "waiting for answers"
    if node == "select_topic":
        return f"topic {update['topic_id']!r}" if update.get("topic_id") else "topic given at start"
    if node == "recommend_videos" and "recommendations" in update:
        r = update["recommendations"]
        return (f"{r.get('strategy_id') or '-'}: {len(r.get('items', []))} videos, status {r.get('status')}"
                + (f" ({'; '.join(r['errors'])[:160]})" if r.get("errors") else ""))
    if node == "select_video":
        action = update.get("video_action")
        if action == "video":
            v = update.get("video") or {}
            return f"chose {v.get('title')!r}" + (" (index reused)" if v.get("reused_index") else "")
        return {"refresh": "asked for new recommendations", "end": "ended the session"}.get(action, "-")
    if node == "prepare_knowledge":
        if update.get("resource_id"):
            v = update.get("video") or {}
            return (f"indexed {v.get('n_chunks', '?')} chunks from {v.get('title')!r} "
                    f"({v.get('caption_type')} captions, {v.get('caption_language')}) → {update['resource_id']}")
        return update.get("knowledge_error") or "-"
    if node == "collect_context" and "context" in update:
        c = update["context"]
        return f"rtt={c['rtt_ms']:.0f} ms, cpu={c['cpu']:.0%}, mem={c['mem']:.0%}, last_qoe={c['last_qoe']}"
    if node == "select_policy":
        return f"{update.get('policy_id')}: {update.get('decision_reason')}"
    if node == "ask_question":
        if update.get("continue_session") is False:
            return "wants another video" if update.get("video_action") == "more" else "post-test requested"
        return f"query={update.get('query')!r}"
    if node == "crag_retrieve":
        if "retrieval_score" in update:
            return f"best score={update['retrieval_score']:.3f}, chunks kept={len(update['chunks'])}"
        return "; ".join(update.get("q_errors", [])[-1:]) or "-"
    if node == "corrective_retrieve":
        return f"retry {update.get('crag_retries')}: {update.get('retrieval_query')!r}"
    if node == "llm_router":
        esc = update.get("escalation") or {}
        return str(update.get("model_used", "")) + (f" (escalated: {esc['reason']})" if esc.get("reason") else "")
    if node == "generate_response" and "model_used" in update:
        return f"answered by {update['model_used']}"
    if node == "post_test_and_feedback":
        return f"post_score={update.get('post_score')}, rating={update.get('rating')}"
    if node == "evaluate_session" and "reward" in update:
        return f"QoE={update['qoe']:.2f}, SRE={update['sre']:.2f}, reward={update['reward']:+.3f}"
    if node == "plan_session":
        plan = update.get("session_plan") or {}
        goals = ", ".join(g["concept_id"] for g in plan.get("goals") or [])
        return f"goals: {goals}" if goals else f"no plan ({plan.get('reason') or plan.get('status')})"
    if node == "update_tracker" and update.get("tracker_updates"):
        last = update["tracker_updates"][-1]
        return f"{last['status']}" + (f": {last['concept_id']} ({last['confidence']})" if last.get("concept_id") else "")
    if node == "monitor_progress" and update.get("progress"):
        return f"{update['progress'].get('touched', 0)}/{update['progress'].get('total', 0)} goals worked on"
    if node == "store_experience":
        return "; ".join(update.get("session_errors", [])) or "experience stored"
    return ", ".join(sorted(update)) or "-"


class AgentBackend:
    """One compiled graph + real services, shared by every Streamlit session in this process."""

    def __init__(self) -> None:
        self.services: Services = get_services()
        self.services.app.ensure_ready()  # create tables if missing, seed policies P1-P5
        self.checkpointer = get_checkpointer()
        self.runner = SessionRunner(build_graph(self.checkpointer))
        # The runner and its SQLite checkpointer are shared; serialise graph execution.
        self._lock = threading.Lock()
        from agent.warmup import start_background_warmup

        start_background_warmup(self.services)  # load models + missing question banks without blocking the UI

    # ---- Sessions ----------------------------------------------------------------------------

    def _run(self, call: Callable[[], dict[str, Any]]) -> StepResult:
        """Run a runner call under the lock, recording every node it executes."""
        trace: list[dict[str, Any]] = []

        def on_node(node: str, update: dict[str, Any]) -> None:
            trace.append({"step": len(trace) + 1, "node": node, "detail": describe_node(node, update or {})})

        with self._lock:
            self.runner.on_node = on_node
            started = time.perf_counter()
            try:
                out = call()
                return StepResult(out, trace, (time.perf_counter() - started) * 1000)
            except Exception as exc:  # surfaced verbatim in the UI, never replaced with output
                return StepResult(None, trace, (time.perf_counter() - started) * 1000,
                                  f"{type(exc).__name__}: {exc}", traceback.format_exc())
            finally:
                self.runner.on_node = None

    def start_session(self, student_id: str, topic_id: str = "") -> StepResult:
        """Create the LearningSession row and run the graph to its first interrupt (select_topic)."""
        return self._run(lambda: self.runner.start_session(student_id, topic_id))

    def resume(self, thread_id: str, payload: Any) -> StepResult:
        """Answer the pending interrupt and run to the next one (or the end)."""
        return self._run(lambda: self.runner.resume(thread_id, payload))

    def resume_streaming(self, thread_id: str, payload: Any, on_text: Callable[[str | None], None]) -> StepResult:
        """Like resume, sending the answer text to `on_text` while the model writes it (None = discard so far)."""
        from services.streaming import streaming_to

        with streaming_to(on_text):
            return self.resume(thread_id, payload)

    def get_state(self, thread_id: str) -> dict[str, Any] | None:
        """Return the checkpointed session state, or None if no checkpoint exists for the id."""
        with self._lock:
            try:
                return self.runner.get_state(thread_id)
            except SessionNotFound:
                return None

    def agent_session_count(self) -> int:
        """Number of distinct sessions with checkpoints in data/checkpoints.db."""
        with self._lock:
            self.checkpointer.setup()  # creates the checkpoint tables if nothing was written yet
            return self.checkpointer.conn.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints").fetchone()[0]

    def conversation(self, thread_id: str) -> list[dict[str, Any]]:
        """Rebuild every answered question from the checkpoint history.

        qa_log entries carry no answer text and `response` holds only the latest answer, but
        generate_response writes the answer, its chunks and the qa_log entry in one update, so
        any checkpoint with N log entries and a non-empty response holds answer N.
        """
        cfg = {"configurable": {"thread_id": thread_id}}
        answers: dict[int, dict[str, Any]] = {}
        with self._lock:
            history = list(self.runner.graph.get_state_history(cfg))
        for snapshot in history:
            values = snapshot.values or {}
            n = len(values.get("qa_log", []))
            if n and values.get("response") and n not in answers:
                answers[n] = {"response": values["response"], "chunks": values.get("chunks", [])}
        log = history[0].values.get("qa_log", []) if history else []
        return [{"log": entry, **answers.get(i, {"response": None, "chunks": []})}
                for i, entry in enumerate(log, start=1)]

    # ---- LLM routes --------------------------------------------------------------------------

    def provider_statuses(self) -> list[ProviderStatus]:
        """Ask the agent's router whether each route is usable (health cached for LLM_HEALTH_TTL_S)."""
        router = self.services.router
        how = {
            "local": f"GET {settings.ollama_base_url}/api/tags lists the model",
            "cloud": f"{getattr(self.services.router.clients.get('cloud'), 'label', 'cloud')} API key is configured "
                     "(GROQ_API_KEY or HF_TOKEN; the router does not probe the network, to save quota)",
        }
        return [ProviderStatus(p, router.model_name(p), router.is_available(p), how[p]) for p in ROUTES]

    # ---- Knowledge / retrieval ----------------------------------------------------------------

    @property
    def embedder_name(self) -> str:
        return self.services.crag.store.embedder.name

    @staticmethod
    def upload_dir() -> Path:
        """Where uploaded transcripts are stored (KNOWLEDGE_STORAGE_PATH)."""
        return get_app_settings().resolved_storage_paths()["knowledge"]

    def topics(self) -> dict[str, Path | None]:
        """Topics with a transcript (samples or uploads) or an existing index → transcript path."""
        found: dict[str, Path | None] = {}
        for folder in (settings.knowledge_samples_dir, self.upload_dir()):
            if folder.is_dir():
                for path in sorted(folder.glob("*.txt")):
                    found[path.stem] = path
        if settings.vector_index_dir.is_dir():
            for meta in settings.vector_index_dir.glob("*.json"):
                found.setdefault(meta.stem, None)
        return dict(sorted(found.items(), key=lambda kv: catalog_order(kv[0])))

    def index_status(self, topic: str) -> IndexStatus:
        """Read the topic's index sidecar and check it matches the configured embedder."""
        source = self.topics().get(topic)
        meta_path = settings.vector_index_dir / f"{topic}.json"
        if not self.services.crag.store.exists(topic):
            return IndexStatus(topic, False, source)
        try:
            meta = IndexMeta.model_validate(json.loads(meta_path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            return IndexStatus(topic, True, source, error=f"Unreadable index metadata: {exc}")
        compatible = meta.embedding_model == self.embedder_name
        error = None if compatible else (f"Built with {meta.embedding_model}, but the configured embedder is "
                                         f"{self.embedder_name}. Rebuild the index.")
        return IndexStatus(topic, True, source, meta.embedding_model, len(meta.chunks), meta.dim, compatible, error)

    def build_index(self, topic: str, transcript: Path) -> IndexMeta:
        """Chunk, embed and index a transcript with knowledge_prep.build_index (the project's own builder)."""
        from knowledge_prep.build_index import build_topic_index

        meta = build_topic_index(topic, transcript)
        # The agent's store caches loaded indexes; drop the stale copy so the new one is read.
        self.services.crag.store._cache.pop(topic, None)
        return meta

    def retrieve(self, topic: str, query: str, top_k: int, *, corrective: bool = False) -> RetrievalResult:
        """Run the agent's CRAG retrieval; corrective=True rewrites the query as corrective_retrieve does."""
        crag = self.services.crag
        if corrective:
            return crag.retrieve(topic, crag.rewrite_query(query, topic, 1), top_k, grade_query=query)
        return crag.retrieve(topic, query, top_k)

    # ---- Monitoring --------------------------------------------------------------------------

    def cpu_mem(self) -> tuple[float, float]:
        """psutil CPU and memory utilisation as fractions (services.monitoring.SystemMonitor)."""
        return self.services.monitor.cpu_mem()

    def probe_rtt(self) -> tuple[float | None, str | None]:
        """One HTTP HEAD to PROBE_URL, exactly as collect_context measures RTT."""
        try:
            return self.services.monitor.probe_rtt(settings.probe_url, settings.probe_timeout_s), None
        except ProbeError as exc:
            return None, str(exc)
