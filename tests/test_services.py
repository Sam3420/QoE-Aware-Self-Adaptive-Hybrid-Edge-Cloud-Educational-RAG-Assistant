"""Unit tests for the vector store, CRAG and LLM router (offline: hashing embedder, fake clients)."""
import pytest

from knowledge_prep.build_index import chunk_text
from services.crag import CragRetriever
from services.fakes import FakeLLMClient
from services.llm_clients import ExtractiveClient, GenerationRequest, LLMError, build_request
from services.llm_router import LLMRouter
from services.vector_store import (
    Chunk,
    EmbeddingError,
    FaissVectorStore,
    HashingEmbedder,
    IndexNotFoundError,
    RetrievedChunk,
)

DOCS = [
    "Chlorophyll is the green pigment in leaves that absorbs sunlight.",
    "Stomata are tiny holes in the leaf that let carbon dioxide in.",
    "The French revolution began in 1789 in Paris.",
]


@pytest.fixture
def store(tmp_path):
    s = FaissVectorStore(base_dir=tmp_path, embedder=HashingEmbedder())
    s.build("topic", [Chunk(chunk_id=f"c{i}", text=t) for i, t in enumerate(DOCS)])
    return s


def test_search_ranks_relevant_chunk_first(store):
    hits = store.search("topic", "green pigment chlorophyll", top_k=2)
    assert hits[0].chunk_id == "c0"
    assert hits[0].score >= hits[1].score


def test_search_reloads_index_from_disk(store, tmp_path):
    fresh = FaissVectorStore(base_dir=tmp_path, embedder=HashingEmbedder())
    assert fresh.search("topic", "stomata holes", top_k=1)[0].chunk_id == "c1"


def test_missing_index_and_embedder_mismatch(store, tmp_path):
    with pytest.raises(IndexNotFoundError):
        store.search("other", "x", top_k=1)
    with pytest.raises(EmbeddingError):
        FaissVectorStore(base_dir=tmp_path, embedder=HashingEmbedder(dim=64)).search("topic", "x", top_k=1)


def test_crag_grades_against_original_question(store):
    crag = CragRetriever(store=store, min_chunk_score=0.2)
    # The rewritten query finds the chlorophyll chunk, but the original question is off-topic.
    result = crag.retrieve("topic", "chlorophyll green pigment", top_k=1, grade_query="french revolution paris")
    assert result.top_score < 0.2
    assert result.chunks == []


def test_rewrite_query_anchors_topic_and_drops_filler():
    rewritten = CragRetriever.rewrite_query("what is it?", "plant_biology", attempt=1)
    assert rewritten.startswith("plant biology:")
    assert "what" not in rewritten.split()


def test_merge_keeps_best_score_per_chunk():
    a = [RetrievedChunk(chunk_id="x", text="t", score=0.4), RetrievedChunk(chunk_id="y", text="t", score=0.9)]
    b = [RetrievedChunk(chunk_id="x", text="t", score=0.7)]
    merged = CragRetriever.merge(a, b, top_k=2)
    assert [(c.chunk_id, c.score) for c in merged] == [("y", 0.9), ("x", 0.7)]


def test_chunk_text_overlaps():
    words = " ".join(f"w{i}." for i in range(50))
    chunks = chunk_text(words, resource_id="r", size=20, overlap=5)
    assert len(chunks) > 2
    assert chunks[0].text.split()[-5:] == chunks[1].text.split()[:5]


class _DownClient(FakeLLMClient):
    def is_available(self) -> bool:
        return False

    def generate(self, request: GenerationRequest):
        raise LLMError("down")


def test_router_health_and_failure_marking():
    router = LLMRouter([FakeLLMClient("local", "m", 1.0), _DownClient("cloud", "c", 1.0), ExtractiveClient()])
    assert router.is_available("local") and not router.is_available("cloud")
    request = build_request("q", ["Chunk one. Two."], grounded=True, competency="beginner", language="en",
                            max_tokens=10, temperature=0.0)
    with pytest.raises(LLMError):
        router.generate("cloud", request)
    assert router.generate("extractive", request).text.startswith("From the lesson:")


def test_prompt_expands_bare_topic_and_flags_low_confidence():
    request = build_request("photosynthesis?", ["ctx"], grounded=False, competency="beginner", language="hi",
                            max_tokens=10, temperature=0.0)
    assert "Explain photosynthesis." in request.prompt
    assert "NOT covered" in request.prompt
    assert "Hindi" in request.system


def test_mcq_adapter_sends_schema_and_prefers_cloud():
    from agent.interfaces import RouterLLMProvider
    from app.llm.types import LLMGenerateRequest

    seen = {}

    class Capture(FakeLLMClient):
        def generate(self, request):
            seen[self.provider] = request
            return super().generate(request)

    router = LLMRouter([Capture("local", "l", 1.0), Capture("cloud", "c", 1.0), ExtractiveClient()])
    result = RouterLLMProvider(router).generate(
        LLMGenerateRequest(prompt="p", system_prompt="s", model_id="x", max_tokens=10, temperature=0.0))
    assert result.model_provider == "cloud" and list(seen) == ["cloud"]
    schema = seen["cloud"].json_schema
    assert schema["type"] == "array" and schema["items"]["required"] == ["question", "options", "answer_index", "concept"]
