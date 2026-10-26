"""Langfuse tracing audit (offline): the exported span trees of a learning session and a video
recommendation are checked against the Langfuse trace best practices. Spans go to an in-memory
OpenTelemetry exporter; nothing is sent over the network."""
import json
from collections import defaultdict

import httpx
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent.recommendations import make_recommender
from config import Settings, settings
from services.llm_clients import OllamaClient
from services.vector_store import HashingEmbedder
from telemetry import tracing
from tests.conftest import ANSWERS, pass_pretest, seed_experiences
from tests.youtube_fakes import OBJECTIVE, FakeTranscripts, FakeYouTubeApi, make_client

TYPE, LEVEL = "langfuse.observation.type", "langfuse.observation.level"
INPUT, OUTPUT = "langfuse.observation.input", "langfuse.observation.output"


@pytest.fixture(scope="module")
def langfuse_client():
    """One test client for the module (Langfuse shares OpenTelemetry state per process)."""
    from langfuse import Langfuse

    exporter = InMemorySpanExporter()
    client = Langfuse(public_key="pk-lf-test", secret_key="sk-lf-test", base_url="http://127.0.0.1:9",
                      environment="test", span_exporter=exporter, mask_otel_spans=tracing.mask_otel_spans)
    yield client, exporter
    tracing.reset()


@pytest.fixture
def traced(langfuse_client, monkeypatch):
    """Tracing on, exporter cleared, scores captured instead of sent."""
    client, exporter = langfuse_client
    exporter.clear()
    scores = []
    monkeypatch.setattr(client, "create_score", lambda **kw: scores.append(kw))
    tracing.set_client(client)
    yield client, exporter, scores
    tracing.set_client(None)


def spans_by_trace(client, exporter):
    client.flush()
    traces = defaultdict(list)
    for span in exporter.get_finished_spans():
        traces[format(span.context.trace_id, "032x")].append(span)
    return traces


def root(spans):
    return next(s for s in spans if s.parent is None)


def by_name(spans, name):
    return [s for s in spans if s.name == name]


def load(span, key):
    """Attribute value; JSON payloads (input/output/metadata/usage) are decoded."""
    value = span.attributes.get(key)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


@pytest.fixture
def ollama(services):
    """A real OllamaClient over a mocked HTTP transport, so generations carry real usage fields."""
    def handler(request):
        if request.url.path.endswith("/api/chat"):
            body = json.loads(request.content)
            return httpx.Response(200, json={"message": {"role": "assistant", "content": f"Answer to: {body['messages'][1]['content'][-40:]}"},
                                             "prompt_eval_count": 120, "eval_count": 30})
        return httpx.Response(200, json={"models": [{"name": "gemma3:1b"}]})

    client = OllamaClient(model="gemma3:1b", base_url="http://ollama.test", timeout_s=5, health_timeout_s=1)
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    services.router.clients["local"] = client
    return client


def run_session(runner, learner, comment="clear and helpful"):
    out = runner.start_session(learner, "photosynthesis")
    tid = out["thread_id"]
    pass_pretest(runner, tid)
    runner.resume(tid, {"query": "How do plants make glucose from light?"})
    runner.resume(tid, {"end": True})
    final = runner.resume(tid, {"answers": ANSWERS, "rating": 4, "comment": comment})
    return tid, final


def test_learning_session_traces_follow_best_practices(traced, runner, learner, services, ollama):
    client, exporter, scores = traced
    seed_experiences(services, [("P1", 0.8)], context={"rtt_ms": 180.0, "cpu": 0.35, "mem": 0.55,
                                                        "competency_num": 1.0, "last_qoe": 0.5})
    tid, final = run_session(runner, learner)
    assert final.get("trace_id")  # the runner returns the trace id of each step

    traces = spans_by_trace(client, exporter)
    roots = {root(spans).name: spans for spans in traces.values()}
    # one trace per learner step, verb-first names
    assert set(roots) == {"start-learning-session", "submit-pretest", "choose-video", "answer-question",
                          "end-questions", "submit-posttest"}
    for spans in roots.values():
        assert load(root(spans), TYPE) == "agent"
        assert all(s.attributes.get("session.id") == tid and s.attributes.get("user.id") == learner for s in spans)
        assert all(s.attributes.get("langfuse.trace.tags") == ("learning-session",) for s in spans)
        # nested helpers (recommend-videos) must not rename the step's trace
        assert {s.attributes.get("langfuse.trace.name") for s in spans} <= {root(spans).name, None}
        assert not [s.name for s in spans if s.attributes.get(LEVEL) == "ERROR"]  # interrupts are not errors
        assert all(s.attributes.get(INPUT) is not None or s.attributes.get(OUTPUT) is not None for s in spans)

    answer = roots["answer-question"]
    assert load(root(answer), INPUT) == "How do plants make glucose from light?"  # readable root I/O
    assert load(root(answer), OUTPUT).startswith("Answer to:")
    gen = by_name(answer, "generate-answer")[0]
    assert load(gen, TYPE) == "generation" and gen.attributes["langfuse.observation.model.name"] == "gemma3:1b"
    assert load(gen, "langfuse.observation.usage_details") == {"input": 120, "output": 30}
    assert [m["role"] for m in load(gen, INPUT)] == ["system", "user"]  # OpenAI-format messages
    retr = by_name(answer, "retrieve-lesson-context")[0]
    assert load(retr, TYPE) == "retriever" and load(retr, OUTPUT)["chunks"]
    ids = {s.context.span_id for s in answer}
    assert all(s.parent.span_id in ids for s in answer if s.parent)  # nothing dangles outside the tree

    start = roots["start-learning-session"]
    assert load(by_name(start, "choose-topic")[0], OUTPUT) == {"topic_id": "preset"}
    assert load(by_name(start, "plan-pretest")[0], OUTPUT)["items"] > 0
    assert load(by_name(start, "run-pretest")[0], OUTPUT) == {"waiting_for_learner": "pre_test", "reask_error": None}
    pretest = roots["submit-pretest"]
    assert load(by_name(pretest, "receive-video-choice")[0], OUTPUT)["waiting_for_learner"] == "select_video"
    policy = roots["choose-video"]
    chosen = load(by_name(policy, "receive-video-choice")[0], OUTPUT)
    assert chosen["action"] == "video" and chosen["video_id"]
    assert load(by_name(policy, "index-video-transcript")[0], OUTPUT)["resource_id"]
    assert load(by_name(policy, "select-policy")[0], TYPE) == "chain"
    assert load(by_name(policy, "retrieve-similar-sessions")[0], TYPE) == "retriever"
    assert load(by_name(policy, "measure-context")[0], TYPE) == "tool"
    assert load(by_name(roots["submit-posttest"], "evaluate-session")[0], TYPE) == "evaluator"

    names = {(s["name"], s.get("session_id")) for s in scores}
    assert {("session_rating", tid), ("posttest_score", tid), ("session_qoe", tid), ("session_sre", tid),
            ("adaptation_reward", tid)} <= names
    assert all(s["data_type"] == "NUMERIC" for s in scores)


def test_watching_another_video_is_its_own_step_and_retrieval_names_the_video(traced, runner, learner, services,
                                                                             ollama):
    client, exporter, _ = traced
    tid = runner.start_session(learner, "photosynthesis")["thread_id"]
    first = pass_pretest(runner, tid)["interrupt"]["video"]
    runner.resume(tid, {"query": "How do plants make glucose from light?"})
    pause = runner.resume(tid, {"more_videos": True})["interrupt"]
    runner.resume(tid, {"item_id": next(i["item_id"] for i in pause["items"] if not i["watched"])})
    runner.resume(tid, {"query": "What does chlorophyll absorb?"})

    traces = spans_by_trace(client, exporter)
    roots = {}
    for spans in traces.values():
        roots.setdefault(root(spans).name, []).append(spans)
    more = roots["watch-another-video"][0]
    assert load(root(more), INPUT) == {"more_videos": True}
    assert load(root(more), OUTPUT)["next"] == "select_video"
    assert load(by_name(more, "receive-question")[0], OUTPUT) == {"next": "watch-another-video"}
    assert load(by_name(more, "receive-video-choice")[0], INPUT)["already_watched"] == [first["video_id"]]

    first_q, second_q = roots["answer-question"]
    inputs = [load(by_name(t, "retrieve-lesson-context")[0], INPUT) for t in (first_q, second_q)]
    assert inputs[0]["resource_id"] == f"yt_{first['video_id']}" and inputs[0]["video_title"] == first["title"]
    assert inputs[1]["resource_id"] != inputs[0]["resource_id"] and inputs[1]["resource_id"].startswith("yt_")
    assert sorted(load(root(t), INPUT) for t in roots["answer-question"]) == [
        "How do plants make glucose from light?", "What does chlorophyll absorb?"]  # no empty question traces


def test_failed_llm_call_is_an_error_generation_and_fallback_is_visible(traced, runner, learner, services):
    client, exporter, _ = traced

    def handler(request):
        if request.url.path.endswith("/api/chat"):
            return httpx.Response(503, json={"error": "model loading"})
        return httpx.Response(200, json={"models": [{"name": "gemma3:1b"}]})

    broken = OllamaClient(model="gemma3:1b", base_url="http://ollama.test", timeout_s=5, health_timeout_s=1)
    broken._http = httpx.Client(transport=httpx.MockTransport(handler))
    services.router.clients["local"] = broken
    out = runner.start_session(learner, "photosynthesis")
    pass_pretest(runner, out["thread_id"])
    runner.resume(out["thread_id"], {"query": "How do plants make glucose from light?"})
    answer = next(spans for spans in spans_by_trace(client, exporter).values()
                  if root(spans).name == "answer-question")
    failed = [s for s in by_name(answer, "generate-answer") if s.attributes.get(LEVEL) == "ERROR"]
    assert len(failed) == 1 and "503" in failed[0].attributes["langfuse.observation.status_message"]
    assert failed[0].attributes["langfuse.observation.model.name"] == "gemma3:1b"
    # The run recovered (fallback route): the root is not an error and still has the learner's answer.
    assert root(answer).attributes.get(LEVEL) != "ERROR" and load(root(answer), OUTPUT)


def test_pii_is_redacted_and_content_capture_can_be_disabled(traced, runner, learner, services, ollama,
                                                             monkeypatch):
    client, exporter, _ = traced
    run_session(runner, learner, comment="call me on 9876543210 or mail kid@example.com")
    posts = by_name([s for spans in spans_by_trace(client, exporter).values() for s in spans], "score-posttest")
    # the node is traced twice: once waiting for the learner, once with the submitted answers
    comment = next(load(s, OUTPUT)["comment"] for s in posts if "comment" in (load(s, OUTPUT) or {}))
    assert "[PHONE]" in comment and "[EMAIL]" in comment and "9876543210" not in comment
    assert tracing.redact(learner) == learner  # ids are not mistaken for phone/id numbers

    exporter.clear()
    monkeypatch.setattr(settings, "langfuse_capture_content", False)
    out = runner.start_session(learner, "photosynthesis")
    pass_pretest(runner, out["thread_id"])
    runner.resume(out["thread_id"], {"query": "How do plants make glucose from light?"})
    answer = next(spans for spans in spans_by_trace(client, exporter).values()
                  if root(spans).name == "answer-question")
    assert load(root(answer), INPUT) == "[content capture disabled]"
    gen = by_name(answer, "generate-answer")[0]
    assert load(gen, INPUT) == "[content capture disabled]"
    assert load(gen, "langfuse.observation.usage_details") == {"input": 120, "output": 30}  # numbers still sent


def test_recommendation_trace_keeps_threaded_transcripts_and_scores_feedback(traced, services, learner, tmp_path):
    client, exporter, scores = traced
    cfg = Settings(_env_file=None, rec_similarity_floor=0.0, rec_similarity_ceil=0.6, youtube_cache_dir=tmp_path,
                   transcript_chunk_words=20, transcript_chunk_overlap_words=5, rec_concept_cover_threshold=0.5)
    rec = make_recommender(services, data_client=make_client(FakeYouTubeApi(), api_key="SECRET-KEY"),
                           transcripts=FakeTranscripts(), embedder=HashingEmbedder(), cfg=cfg)
    run = rec.recommend(learner, "photosynthesis", objective=OBJECTIVE)
    traces = spans_by_trace(client, exporter)
    assert len(traces) == 1  # transcript fetches in worker threads did not start orphan traces
    spans = next(iter(traces.values()))
    r = root(spans)
    assert r.name == "recommend-videos" and load(r, TYPE) == "agent"
    assert load(r, OUTPUT)["recommendations"][0]["video_id"] == "goodTeach01"
    assert {s.name for s in spans} >= {"select-strategy", "search-youtube", "fetch-video-metadata",
                                       "fetch-transcript", "rank-videos"}
    assert len(by_name(spans, "fetch-transcript")) == 4
    assert all(s.attributes.get("user.id") == learner for s in spans)
    exported = json.dumps([dict(s.attributes) for s in spans], default=str)
    assert "SECRET-KEY" not in exported  # the API key never reaches a payload
    no_transcript = [s for s in by_name(spans, "fetch-transcript") if load(s, OUTPUT)["status"] != "ok"]
    assert no_transcript and all(s.attributes[LEVEL] == "WARNING" for s in no_transcript)

    item = run["recommendations"][0]["item_id"]
    rec.record_event(run["id"], "accepted", item_id=item)
    rec.record_event(run["id"], "rated", item_id=item, value={"rating": 5})
    rec.record_event(run["id"], "displayed", item_id=item)  # not a judgement: no score
    trace_id = format(r.context.trace_id, "032x")
    assert [(s["name"], s["value"], s["trace_id"]) for s in scores] == [
        ("video_marked_helpful", 1, trace_id), ("video_rating", 5, trace_id)]

    # Called from a graph node (inside a learner-step trace) it nests instead of starting or renaming a trace.
    exporter.clear()
    with tracing.trace("submit-pretest", user_id=learner, session_id="s-1", tags=["learning-session"]):
        rec.recommend(learner, "photosynthesis", objective=OBJECTIVE)
    traces = spans_by_trace(client, exporter)
    assert len(traces) == 1
    spans = next(iter(traces.values()))
    assert root(spans).name == "submit-pretest" and by_name(spans, "recommend-videos")[0].parent is not None
    assert {s.attributes.get("langfuse.trace.name") for s in spans} <= {"submit-pretest", None}
    assert all(s.attributes.get("session.id") == "s-1" for s in spans)


def test_tracing_off_or_broken_never_affects_the_app(runner, learner, services, ollama):
    tracing.set_client(None)
    with tracing.observation("x") as obs:
        obs.update(output=1)
    _, final = run_session(runner, learner)
    assert final["final_state"]["evaluation"]["status"] == "completed" and "trace_id" not in final

    class Broken:
        def start_as_current_observation(self, **_):
            raise RuntimeError("langfuse down")

        def get_current_trace_id(self):
            raise RuntimeError("langfuse down")

        def create_score(self, **_):
            raise RuntimeError("langfuse down")

    tracing.set_client(Broken())
    try:
        _, final = run_session(runner, learner)
        assert final["final_state"]["evaluation"]["status"] == "completed"
    finally:
        tracing.set_client(None)


def test_speech_traces_join_the_learners_session_with_load_and_latency(traced):
    """Mic questions and read-aloud are their own traces in the learner's Langfuse session, with the model load
    (first use) and the inference as child spans, and latency / real-time factor in the output."""
    from types import SimpleNamespace as NS

    import numpy as np

    from services.speech.audio import AudioError, encode_wav
    from services.speech.stt import WhisperSTT
    from services.speech.tts import KokoroTTS

    client, exporter, _ = traced
    speech_wav = encode_wav((0.5 * np.sin(np.arange(16000) / 5)).astype(np.float32), 16000)
    stt = WhisperSTT(model_id="distil-whisper/distil-small.en")

    def fake_load():
        stt._pipe, stt.device_used = (lambda inputs: {"text": "What is photosynthesis?"}), "cpu"
        return 1.5

    stt.load = fake_load
    stt.transcribe(speech_wav, user_id="learner-1", session_id="session-1")
    stt.transcribe(speech_wav, user_id="learner-1", session_id="session-1")  # model already loaded
    tts = KokoroTTS(repo_id="onnx-community/Kokoro-82M-v1.0-ONNX", model_file="onnx/model_quantized.onnx",
                    engine=NS(create=lambda *a, **k: (np.zeros(12000, dtype=np.float32), 24000)))
    tts.synthesize("**Plants** make food.", user_id="learner-1", session_id="session-1")
    with pytest.raises(AudioError):
        stt.transcribe(b"not audio", user_id="learner-1", session_id="session-1")

    traces = sorted(spans_by_trace(client, exporter).values(), key=lambda s: root(s).start_time)
    assert [root(s).name for s in traces] == ["transcribe-question", "transcribe-question", "read-aloud",
                                              "transcribe-question"]
    for spans in traces:
        assert all(s.attributes.get("session.id") == "session-1" and s.attributes.get("user.id") == "learner-1"
                   for s in spans)
    first, second, read, failed = traces
    assert {s.name for s in first} == {"transcribe-question", "load-stt-model", "transcribe-speech"}
    assert load(by_name(first, "load-stt-model")[0], OUTPUT) == {"seconds": 1.5, "device": "cpu"}
    assert {s.name for s in second} == {"transcribe-question", "transcribe-speech"}  # no reload
    out = load(root(second), OUTPUT)
    assert out["text"] == "What is photosynthesis?" and out["audio_s"] == 1.0 and "real_time_factor" in out
    assert load(root(read), OUTPUT)["audio_s"] == 0.5 and by_name(read, "synthesize-speech")
    assert root(failed).attributes.get(LEVEL) == "ERROR"  # an unreadable recording is visible as an error
