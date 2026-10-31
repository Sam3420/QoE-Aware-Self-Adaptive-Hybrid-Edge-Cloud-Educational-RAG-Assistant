"""Latency features: Ollama keep_alive and streaming, the token sink reaching the graph, index locks and prefetch,
and the background warm-up (all offline: mocked HTTP, fake services)."""
import json
import threading
import time

import httpx
import pytest

from agent import knowledge
from agent.warmup import warm_up
from config import Settings
from services.llm_clients import GenerationRequest, LLMError, OllamaClient
from services.streaming import streaming_to
from tests.conftest import pass_pretest


def ollama(handler, keep_alive="30m"):
    client = OllamaClient(model="gemma3:1b", base_url="http://ollama.test", timeout_s=5, health_timeout_s=1,
                          keep_alive=keep_alive)
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def chat_handler(bodies, pieces=("Plants ", "make ", "glucose."), fail_after=None):
    """/api/chat: NDJSON chunks when streaming (optionally a broken stream), one JSON reply otherwise."""
    def handler(request):
        if request.url.path.endswith("/api/tags"):
            return httpx.Response(200, json={"models": [{"name": "gemma3:1b"}]})
        body = json.loads(request.content)
        bodies.append(body)
        if not body["stream"]:
            return httpx.Response(200, json={"message": {"content": "".join(pieces)}, "eval_count": 3})
        lines = [{"message": {"content": p}, "done": False} for p in pieces]
        if fail_after is not None:
            lines = lines[:fail_after] + [{"error": "model crashed"}]
        else:
            lines.append({"message": {"content": ""}, "done": True, "prompt_eval_count": 40, "eval_count": 3})
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())
    return handler


def request(**kw):
    return GenerationRequest(system="s", prompt="p", max_tokens=50, temperature=0.2, **kw)


def test_keep_alive_is_sent_and_no_stream_without_a_sink():
    bodies = []
    result = ollama(chat_handler(bodies)).generate(request())
    assert result.text == "Plants make glucose." and bodies[0]["keep_alive"] == "30m" and bodies[0]["stream"] is False
    bodies.clear()
    ollama(chat_handler(bodies), keep_alive=None).generate(request())
    assert "keep_alive" not in bodies[0]  # Ollama's own default


def test_answer_streams_to_the_sink():
    bodies, got = [], []
    with streaming_to(got.append):
        result = ollama(chat_handler(bodies)).generate(request())
    assert bodies[0]["stream"] is True and got == ["Plants ", "make ", "glucose."]
    assert result.text == "Plants make glucose." and result.completion_tokens == 3


def test_structured_output_never_streams():
    bodies, got = [], []
    with streaming_to(got.append):
        ollama(chat_handler(bodies)).generate(request(json_schema={"type": "object"}))
    assert bodies[0]["stream"] is False and got == []  # MCQ JSON is not shown to the learner


def test_failed_stream_tells_the_sink_to_discard_partial_text():
    got = []
    with streaming_to(got.append), pytest.raises(LLMError, match="model crashed"):
        ollama(chat_handler([], fail_after=1)).generate(request())
    assert got == ["Plants ", None]


def test_warm_up_loads_the_model_with_keep_alive():
    bodies = []
    ollama(chat_handler(bodies)).warm_up()
    assert bodies[0]["options"] == {"num_predict": 1} and bodies[0]["keep_alive"] == "30m"


def test_sink_reaches_the_answer_node_through_the_graph(runner, learner, services):
    bodies, got = [], []
    services.router.clients["local"] = ollama(chat_handler(bodies))
    tid = runner.start_session(learner, "photosynthesis")["thread_id"]
    pass_pretest(runner, tid)
    with streaming_to(got.append):
        out = runner.resume(tid, {"query": "How do plants make glucose from light?"})
    values = runner.get_state(tid)["values"]
    assert "".join(p for p in got if p) == "Plants make glucose." and "glucose" in values["response"]
    assert values["qa_log"][-1]["model_used"] == "local:gemma3:1b"
    assert out["interrupt"]["type"] == "ask_question"
    got.clear()
    runner.resume(tid, {"query": "Why are leaves green?"})  # outside the block: nothing streams
    assert got == []


class SlowStore:
    """exists()/build() store that counts builds; build takes a moment so threads overlap."""

    def __init__(self):
        self.built, self.builds = set(), 0

    def exists(self, resource_id):
        return resource_id in self.built

    def build(self, resource_id, chunks):
        time.sleep(0.2)
        self.builds += 1
        self.built.add(resource_id)
        return type("Meta", (), {"chunks": chunks, "embedding_model": "fake"})()


def test_concurrent_builds_of_one_index_run_once(monkeypatch):
    store = SlowStore()
    monkeypatch.setattr(knowledge, "index_video",
                        lambda store, transcripts, video, languages: (store.build(f"yt_{video['video_id']}", []),
                                                                       {"resource_id": f"yt_{video['video_id']}"})[1])
    results = []
    threads = [threading.Thread(target=lambda: results.append(
        knowledge.ensure_video_index(store, None, {"video_id": "v1"}, ["en"]))) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.builds == 1 and sorted(r["reused_index"] for r in results) == [False, True, True]


def test_prefetch_indexes_in_background_and_skips_ready_videos(monkeypatch):
    store = SlowStore()
    store.built.add("yt_ready")
    built = threading.Event()

    def fake_index(store, transcripts, video, languages):
        store.build(f"yt_{video['video_id']}", [])
        built.set()
        return {"resource_id": f"yt_{video['video_id']}"}

    monkeypatch.setattr(knowledge, "index_video", fake_index)
    started = time.perf_counter()
    queued = knowledge.prefetch_video_indexes(store, None, [{"video_id": "ready"}, {"video_id": "new"}], ["en"])
    assert queued == 1 and time.perf_counter() - started < 0.1  # returns at once
    assert built.wait(5) and store.exists("yt_new")
    # the learner picks the prefetched video: reused, not rebuilt
    assert knowledge.ensure_video_index(store, None, {"video_id": "new"}, ["en"])["reused_index"] is True
    assert store.builds == 1


def test_warm_up_generates_missing_question_banks_once(services):
    cfg = Settings(_env_file=None, prewarm_question_banks=True, prewarm_stt=False,
                   prewarm_video_caches=False)
    first = warm_up(services, cfg)
    assert first["banks_generated"] and not [e for e in first["errors"] if "Error" in e]
    assert warm_up(services, cfg)["banks_generated"] == []  # already stored: nothing to do
