"""Deterministic offline stand-ins for the vector store and LLM clients (stub runs and tests)."""
import time

from services.llm_clients import GenerationRequest, GenerationResult
from services.vector_store import RetrievedChunk

VAGUE_QUERY_WORDS = 3


class FakeSearchBackend:
    """Vague queries (≤3 words) find weak chunks, specific ones strong chunks, so corrective rewrites help.

    Each chunk has a fixed relevance, so re-grading against the original question keeps its score.
    """

    def _chunk(self, resource_id: str, kind: str, i: int) -> RetrievedChunk:
        """Build one fake chunk; 'weak' chunks score 0.55, 'strong' ones 0.85 (minus rank)."""
        base = 0.55 if kind == "weak" else 0.85
        return RetrievedChunk(chunk_id=f"{resource_id}-{kind}-{i}", text=f"Fake {kind} chunk {i} about {resource_id}.",
                              source="fake", score=round(max(0.0, base - 0.05 * i), 3))

    def search(self, resource_id: str, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return top_k fake chunks with decreasing scores."""
        kind = "weak" if len(query.split()) <= VAGUE_QUERY_WORDS else "strong"
        return [self._chunk(resource_id, kind, i) for i in range(top_k)]

    def score_chunks(self, resource_id: str, query: str, chunk_ids: list[str]) -> dict[str, float]:
        """Return each chunk's fixed relevance."""
        out = {}
        for cid in chunk_ids:
            kind, i = cid.rsplit("-", 2)[1:]
            out[cid] = self._chunk(resource_id, kind, int(i)).score
        return out


class FakeLLMClient:
    """Always-available client returning a canned answer with a fixed latency."""

    def __init__(self, provider: str, model: str, latency_ms: float) -> None:
        self.provider = provider
        self.model = model
        self.latency_ms = latency_ms

    def is_available(self) -> bool:
        """Always available."""
        return True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return a canned answer quoting the question."""
        question = request.prompt.rsplit("Question:", 1)[-1].split("\n")[0].strip()
        return GenerationResult(text=f"[{self.model}] Stub answer to '{question}'.",
                                provider=self.provider, model=self.model, latency_ms=self.latency_ms)


class FakeMonitor:
    """Fixed CPU/memory/RTT; set rtt_ms=None to make the probe fail."""

    def __init__(self, cpu: float = 0.35, mem: float = 0.55, rtt_ms: float | None = 180.0) -> None:
        self.cpu, self.mem, self.rtt_ms = cpu, mem, rtt_ms

    def cpu_mem(self) -> tuple[float, float]:
        """Return the fixed load."""
        return self.cpu, self.mem

    def probe_rtt(self, url: str, timeout_s: float) -> float:
        """Return the fixed RTT, or raise ProbeError when rtt_ms is None."""
        from services.monitoring import ProbeError

        if self.rtt_ms is None:
            raise ProbeError(f"fake probe to {url} failed")
        return self.rtt_ms
