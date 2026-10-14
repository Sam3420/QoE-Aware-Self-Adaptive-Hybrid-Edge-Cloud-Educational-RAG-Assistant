"""Service container the nodes use; swap it with set_services() for stub runs and tests."""
import random
from dataclasses import dataclass, field
from typing import Any

from agent.interfaces import (
    AppGateway,
    CragKnowledgeRetriever,
    FakeMCQProvider,
    RouterLLMProvider,
    make_app_gateway,
    make_memory_gateway,
)
from config import Settings, settings
from services.crag import CragRetriever
from services.fakes import FakeLLMClient, FakeMonitor, FakeSearchBackend, FakeVideoRecommender
from services.llm_clients import ExtractiveClient, HuggingFaceClient, OllamaClient
from services.llm_router import LLMRouter
from services.monitoring import SystemMonitor
from services.vector_store import Embedder, FaissVectorStore, HashingEmbedder, OllamaEmbedder


@dataclass
class Services:
    """Everything the nodes call into."""

    crag: CragRetriever
    router: LLMRouter
    monitor: Any  # SystemMonitor-like: cpu_mem(), probe_rtt(url, timeout_s)
    app: AppGateway  # Person B's DB and services
    rng: random.Random = field(default_factory=lambda: random.Random(settings.policy_rng_seed))
    experiences: Any = None  # agent.experience_store.ExperienceStore (built lazily from app's DB)
    videos: Any = None  # agent.recommendations.RecommendationCoordinator (built lazily; YouTube + transcripts)

    def video_recommender(self) -> Any:
        """The YouTube recommender (strategy kNN, ranking, stored runs) and its transcript source."""
        if self.videos is None:
            from agent.recommendations import make_recommender

            self.videos = make_recommender(self)
        return self.videos

    def experience_store(self) -> Any:
        """The FAISS-backed experience store over this container's database."""
        if self.experiences is None:
            from agent.experience_store import ExperienceStore

            self.experiences = ExperienceStore(self.app.session_factory)
        return self.experiences


def make_embedder(cfg: Settings = settings) -> Embedder:
    """Build the configured embedder (the same one must build and query an index)."""
    if cfg.embedding_backend == "hashing":
        return HashingEmbedder()
    return OllamaEmbedder(model=cfg.embedding_model, base_url=cfg.ollama_base_url)


def make_real_services(cfg: Settings = settings) -> Services:
    """Wire FAISS-backed CRAG, the local/cloud/extractive router, psutil monitoring and the app DB."""
    store = FaissVectorStore(base_dir=cfg.vector_index_dir, embedder=make_embedder(cfg))
    crag = CragRetriever(store=store, min_chunk_score=cfg.crag_min_chunk_score)
    router = LLMRouter(
        [
            OllamaClient(model=cfg.local_llm_model, base_url=cfg.ollama_base_url,
                         timeout_s=cfg.local_llm_timeout_s, health_timeout_s=cfg.llm_health_timeout_s),
            HuggingFaceClient(model=cfg.cloud_llm_model, url=cfg.hf_inference_url,
                              token=cfg.hf_token, timeout_s=cfg.cloud_llm_timeout_s),
            ExtractiveClient(),
        ],
        health_ttl_s=cfg.llm_health_ttl_s,
    )
    app = make_app_gateway(retriever=CragKnowledgeRetriever(crag), mcq_llm=RouterLLMProvider(router))
    return Services(crag=crag, router=router, monitor=SystemMonitor(), app=app)


def make_fake_services(cfg: Settings = settings, *, app: AppGateway | None = None,
                       monitor: Any = None, rng: random.Random | None = None) -> Services:
    """Wire deterministic offline fakes behind the real CRAG and router, on an in-memory DB by default."""
    crag = CragRetriever(store=FakeSearchBackend(), min_chunk_score=cfg.crag_min_chunk_score)
    router = LLMRouter(
        [
            FakeLLMClient("local", "local-slm-fake", latency_ms=1200.0),
            FakeLLMClient("cloud", "cloud-llm-fake", latency_ms=800.0),
            ExtractiveClient(),
        ],
        health_ttl_s=cfg.llm_health_ttl_s,
    )
    app = app or make_memory_gateway(retriever=CragKnowledgeRetriever(crag), mcq_llm=FakeMCQProvider())
    from agent.experience_store import ExperienceStore

    return Services(crag=crag, router=router, monitor=monitor or FakeMonitor(), app=app,
                    rng=rng or random.Random(0),
                    experiences=ExperienceStore(app.session_factory, index_dir=None),  # in-memory FAISS
                    videos=FakeVideoRecommender())  # offline: no YouTube calls


_services: Services | None = None


def get_services() -> Services:
    """Return the active services, creating the real ones on first use."""
    global _services
    if _services is None:
        _services = make_real_services()
    return _services


def set_services(services: Services | None) -> None:
    """Override the active services (None resets to the real ones on next use)."""
    global _services
    _services = services
