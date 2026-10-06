"""Session runner: the API Person B's UI/API layer calls to drive the agent through its interrupts.

    start_session(student_id, topic_id) -> {"thread_id", "interrupt"}            (or "final_state")
    resume(thread_id, payload)          -> {"thread_id", "interrupt" | "final_state"}
    get_state(thread_id)                -> {"thread_id", "values", "next", "interrupt", "finished"}

thread_id is the LearningSession id created in B's DB, so the checkpoint thread, the session row
and the stored experience share one id. Interrupt payloads always carry a "type"; see docs/agent_runner.md.
"""
from collections.abc import Callable
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from agent.deps import get_services
from agent.state import initial_state


class SessionNotFound(LookupError):
    """Raised when a thread_id has no checkpoint."""


class SessionNotPaused(RuntimeError):
    """Raised when resume() is called on a session that is not waiting at an interrupt."""


def run_until_pause(
    graph: CompiledStateGraph,
    graph_input: Any,
    cfg: dict[str, Any],
    on_node: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Stream until the next interrupt or END; return (interrupt payload or None, nodes visited)."""
    visited: list[str] = []
    for chunk in graph.stream(graph_input, cfg, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                return update[0].value, visited
            visited.append(node)
            if on_node:
                on_node(node, update or {})
    return None, visited


class SessionRunner:
    """Drives sessions of one compiled graph; every call is keyed by thread_id."""

    def __init__(self, graph: CompiledStateGraph | None = None,
                 on_node: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        if graph is None:
            from agent.graph import build_graph  # imported lazily: opens data/checkpoints.db

            graph = build_graph()
        self.graph = graph
        self.on_node = on_node

    @staticmethod
    def _cfg(thread_id: str) -> dict[str, Any]:
        """LangGraph config for a thread."""
        return {"configurable": {"thread_id": thread_id}}

    def _drive(self, thread_id: str, graph_input: Any) -> dict[str, Any]:
        """Run to the next interrupt or the end and package the result."""
        pause, _ = run_until_pause(self.graph, graph_input, self._cfg(thread_id), self.on_node)
        if pause is not None:
            return {"thread_id": thread_id, "interrupt": pause}
        return {"thread_id": thread_id, "final_state": self.graph.get_state(self._cfg(thread_id)).values}

    def start_session(self, student_id: str, topic_id: str) -> dict[str, Any]:
        """Create the learning session in B's DB and run until the first interrupt."""
        app = get_services().app
        app.ensure_ready()
        session_id = app.create_learning_session(student_id)  # raises LearnerNotFound
        return self._drive(session_id, initial_state(student_id, topic_id, session_id))

    def resume(self, thread_id: str, payload: Any) -> dict[str, Any]:
        """Answer the pending interrupt with `payload` and run until the next one or the end."""
        state = self.get_state(thread_id)
        if state["finished"]:
            raise SessionNotPaused(f"Session {thread_id!r} has already finished.")
        return self._drive(thread_id, Command(resume=payload))

    def get_state(self, thread_id: str) -> dict[str, Any]:
        """Return the session's current values, next nodes and pending interrupt (if any)."""
        snapshot = self.graph.get_state(self._cfg(thread_id))
        if not snapshot.values:
            raise SessionNotFound(f"No session with thread_id {thread_id!r}.")
        interrupts = snapshot.interrupts or ()
        # After a node re-asks (a second interrupt() in the same node), LangGraph 1.2 reports an
        # empty snapshot.next even though that node is still pending, so use the tasks instead.
        pending = list(dict.fromkeys([*snapshot.next, *(t.name for t in snapshot.tasks)]))
        return {
            "thread_id": thread_id,
            "values": snapshot.values,
            "next": pending,
            "interrupt": interrupts[0].value if interrupts else None,
            "finished": not pending and not interrupts,
        }


_default_runner: SessionRunner | None = None


def _runner() -> SessionRunner:
    """Return the process-wide runner (SQLite checkpointer, real services)."""
    global _default_runner
    if _default_runner is None:
        _default_runner = SessionRunner()
    return _default_runner


def start_session(student_id: str, topic_id: str) -> dict[str, Any]:
    """Start a session with the default runner."""
    return _runner().start_session(student_id, topic_id)


def resume(thread_id: str, payload: Any) -> dict[str, Any]:
    """Resume a session with the default runner."""
    return _runner().resume(thread_id, payload)


def get_state(thread_id: str) -> dict[str, Any]:
    """Read a session's state with the default runner."""
    return _runner().get_state(thread_id)
