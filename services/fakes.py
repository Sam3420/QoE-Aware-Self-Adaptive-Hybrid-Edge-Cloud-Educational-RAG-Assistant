"""Deterministic offline stand-ins for the vector store and LLM clients (stub runs and tests)."""
import time
from datetime import datetime, timezone
from types import SimpleNamespace

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

    def __init__(self) -> None:
        self.built: dict[str, list] = {}  # resource_id -> chunks passed to build()

    def exists(self, resource_id: str) -> bool:
        """Topic indexes always exist; video indexes exist once built."""
        return not resource_id.startswith("yt_") or resource_id in self.built

    def build(self, resource_id: str, chunks: list) -> SimpleNamespace:
        """Record the chunks; searches return fake chunks regardless."""
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks.")
        self.built[resource_id] = list(chunks)
        return SimpleNamespace(resource_id=resource_id, chunks=list(chunks), embedding_model="fake", dim=0)

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

    def __init__(self, cpu: float = 0.35, mem: float = 0.55, rtt_ms: float | None = 180.0,
                 net_step_bytes: int = 2_000_000) -> None:
        self.cpu, self.mem, self.rtt_ms = cpu, mem, rtt_ms
        self.net_step_bytes, self._net = net_step_bytes, 0

    def net_bytes(self) -> int | None:
        """Grows by net_step_bytes on each call (None when net_step_bytes < 0)."""
        if self.net_step_bytes < 0:
            return None
        self._net += self.net_step_bytes
        return self._net

    def cpu_mem(self) -> tuple[float, float]:
        """Return the fixed load."""
        return self.cpu, self.mem

    def probe_rtt(self, url: str, timeout_s: float) -> float:
        """Return the fixed RTT, or raise ProbeError when rtt_ms is None."""
        from services.monitoring import ProbeError

        if self.rtt_ms is None:
            raise ProbeError(f"fake probe to {url} failed")
        return self.rtt_ms


class FakeTranscriptFetcher:
    """Offline transcripts: every fake video has one except ids listed in `missing`."""

    def __init__(self, missing: tuple[str, ...] = ()) -> None:
        self.missing = set(missing)
        self.calls: list[str] = []

    def fetch(self, video_id: str, languages):
        from services.youtube.models import TranscriptRecord, TranscriptSegment, TranscriptStatus

        self.calls.append(video_id)
        if video_id in self.missing:
            return TranscriptRecord(video_id=video_id, status=TranscriptStatus.NO_TRANSCRIPT,
                                    error=f"fake: no track in {list(languages)}")
        lines = [f"Fake transcript sentence {i} of video {video_id}." for i in range(40)]
        return TranscriptRecord(video_id=video_id, status=TranscriptStatus.OK, language_code=languages[0],
                                language=languages[0], caption_type="manual", retrieved_at=datetime.now(timezone.utc),
                                segments=[TranscriptSegment(start=10.0 * i, duration=10.0, text=t)
                                          for i, t in enumerate(lines)])

    def available(self, video_id: str, languages):
        from services.youtube.models import TranscriptStatus

        return TranscriptStatus.NO_TRANSCRIPT if video_id in self.missing else TranscriptStatus.OK


class FakeVideoRecommender:
    """Offline stand-in for agent.recommendations.RecommendationCoordinator (stub runs only).

    Returns `n` fake videos per call (new ids on each call) and records events in memory.
    """

    def __init__(self, n: int = 3, transcripts: FakeTranscriptFetcher | None = None) -> None:
        self.n = n
        self.transcripts = transcripts or FakeTranscriptFetcher()
        self.calls: list[dict] = []
        self.events: list[tuple[str, str, str | None]] = []

    def recommend(self, student_id: str, topic_id: str, *, session_id: str | None = None,
                  revision: bool = False, **_: object) -> dict:
        self.calls.append({"student_id": student_id, "topic_id": topic_id, "session_id": session_id,
                           "revision": revision})
        run_id = f"fake-run-{len(self.calls)}"
        recs = [{"item_id": f"{run_id}-item-{i}", "rank": i + 1, "video_id": f"fakeVid{len(self.calls)}{i:03d}",
                 "title": f"Fake video {i + 1} about {topic_id}", "description": "Offline stub video.",
                 "thumbnail_url": None, "channel_title": "Fake Channel", "duration_s": 600, "url":
                 f"https://www.youtube.com/watch?v=fakeVid{len(self.calls)}{i:03d}", "score": round(0.9 - 0.1 * i, 2),
                 "explanation": "Stub recommendation (offline fake).", "evidence_level": "metadata",
                 "caption_language": "en", "caption_type": "manual"} for i in range(self.n)]
        return {"id": run_id, "status": "ok", "errors": [], "strategy_id": "R1",
                "decision_reason": "fake recommender", "query": topic_id, "recommendations": recs}

    def record_event(self, run_id: str, event_type: str, *, item_id: str | None = None,
                     value: dict | None = None) -> dict:
        self.events.append((run_id, event_type, item_id))
        return {"event": {"event_type": event_type}, "outcome": {}}

    def evaluate(self, run_id: str) -> dict:
        return {"metrics": {}, "qoe_signals": {}, "reward": None, "reward_is_placeholder": True}
