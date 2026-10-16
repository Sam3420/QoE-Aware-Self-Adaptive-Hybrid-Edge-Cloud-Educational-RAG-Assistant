"""Runtime policies P1–P5: the fixed, predefined catalogue the agent selects from.

The mapping matches Person B's POLICY_SEED (the DB `policies` table), which the agent reads
at runtime via get_policy. B's seed carries a FLAG asking to seed from this file instead;
until then a test checks the two agree. The CRAG threshold is the one exception: the agent
applies config.crag_retrieval_threshold (0.65, calibrated for nomic-embed-text) rather than
the seed's 0.75, as agreed.
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.schemas import RuntimeConfig
from config import settings


class Policy(BaseModel):
    """One runtime configuration: retrieval depth, CRAG retries and LLM route."""

    model_config = ConfigDict(frozen=True)

    policy_id: str
    name: str
    top_k: int = Field(ge=1, le=20)
    max_retries: int = Field(ge=0, le=5)
    route: Literal["local", "cloud", "escalate"]  # local SLM, cloud LLM, or local with cloud escalation (P5)
    threshold: float = Field(default=settings.crag_retrieval_threshold, ge=0.0, le=1.0)

    def runtime_config(self) -> dict[str, Any]:
        """Return the validated runtime_config dict stored in AgentState."""
        return RuntimeConfig(
            top_k=self.top_k, threshold=self.threshold, max_retries=self.max_retries, route=self.route
        ).model_dump()


# top_k and retries rise together within each route: a "light" and a "thorough" option per route.
POLICIES: dict[str, Policy] = {
    # Default: cheapest, runs on the local SLM, suits low-bandwidth rural settings.
    "P1": Policy(policy_id="P1", name="local-light", top_k=3, max_retries=0, route="local"),
    "P2": Policy(policy_id="P2", name="local-thorough", top_k=5, max_retries=1, route="local"),
    "P3": Policy(policy_id="P3", name="cloud-light", top_k=3, max_retries=0, route="cloud"),
    "P4": Policy(policy_id="P4", name="cloud-thorough", top_k=5, max_retries=1, route="cloud"),
    # Confidence-aware cascade: local SLM by default, cloud LLM only for low-confidence or unsupported answers.
    "P5": Policy(policy_id="P5", name="local-escalate", top_k=5, max_retries=1, route="escalate"),
}

DEFAULT_POLICY_ID = settings.default_policy_id


def get_policy(policy_id: str | None) -> Policy:
    """Return the policy for an id, falling back to the default."""
    return POLICIES.get(policy_id or "", POLICIES[DEFAULT_POLICY_ID])


def runtime_config_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Build runtime_config from a DB policy row, applying the configured CRAG threshold."""
    return RuntimeConfig(top_k=row["top_k"], threshold=settings.crag_retrieval_threshold,
                         max_retries=row["max_retries"], route=row["route"]).model_dump()
