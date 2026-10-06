"""Session-context collection used for policy selection."""
import logging
from typing import Any

from agent.deps import get_services
from agent.schemas import SessionContext
from agent.state import AgentState
from config import settings
from services.monitoring import ProbeError

logger = logging.getLogger(__name__)


def collect_context(state: AgentState) -> dict[str, Any]:
    """Build the context vector: RTT to the cloud LLM, CPU/memory load, competency and last QoE."""
    services = get_services()
    errors: list[str] = []

    try:
        cpu, mem = services.monitor.cpu_mem()
    except Exception as exc:  # monitoring must never stop a session
        cpu, mem = 0.5, 0.5
        errors.append(f"context: cpu/mem unavailable ({exc}); used 0.5")
    try:
        rtt_ms = services.monitor.probe_rtt(settings.probe_url, settings.probe_timeout_s)
    except ProbeError as exc:
        rtt_ms = settings.rtt_fail_ms
        errors.append(f"context: {exc}; used RTT_FAIL_MS={settings.rtt_fail_ms:.0f}")

    last_qoe = services.app.get_last_qoe(state["student_id"])
    context = SessionContext(
        rtt_ms=rtt_ms,
        cpu=cpu,
        mem=mem,
        competency_num=settings.competency_levels.get(state["competency"], 0.0),
        last_qoe=settings.default_last_qoe if last_qoe is None else min(max(float(last_qoe), 0.0), 1.0),
    )
    for err in errors:
        logger.warning(err)
    return {"context": context.model_dump(), "session_errors": errors}
