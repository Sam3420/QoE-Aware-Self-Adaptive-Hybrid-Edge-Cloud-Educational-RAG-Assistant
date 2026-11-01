"""Answer one benchmark question with one policy/variant through the agent's own QA nodes, and measure it.

The nodes (crag_retrieve → crag_ok/corrective_retrieve → llm_router → generate_response, including P5's grounding
check and escalation) run unchanged on a minimal state: the same code a learner's question goes through, without the
session around it. Questions are the content packages' multiple-choice items (pre-test bank + parallel post-test), so
answer accuracy is scored against the answer key with no judge and no learner.

Needs isolation.isolate() to have run (it is called by the run_* scripts before importing this module).
"""
import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

from agent.deps import get_services
from agent.nodes import qa
from agent.policies import POLICIES
from agent.routing import crag_ok
from app.core.config import get_settings as app_settings
from config import settings
from knowledge_prep.catalog import TOPICS

TOPIC_IDS = [t.topic_id for t in TOPICS]
LETTERS = "ABCD"


# ---- questions ------------------------------------------------------------------------------------------------

def load_questions(kinds: tuple[str, ...] = ("pre", "post")) -> list[dict[str, Any]]:
    """The class-10 content-package items: pre-test bank (topic_question_banks) and parallel post-test items."""
    db = app_settings().database_path
    out: list[dict[str, Any]] = []
    with sqlite3.connect(db) as con:
        if "pre" in kinds:
            for topic_id, items in con.execute("SELECT topic_id, items FROM topic_question_banks "
                                               "WHERE model_provider = 'content-package'"):
                out += [_item(topic_id, "pre", it.get("id") or f"{topic_id}-pre-{i}", it)
                        for i, it in enumerate(json.loads(items))]
        if "post" in kinds:
            for qid, topic_id, item in con.execute("SELECT id, topic_id, item FROM parallel_test_items"):
                out.append(_item(topic_id, "post", qid, json.loads(item)))
    return sorted((q for q in out if q["topic_id"] in TOPIC_IDS), key=lambda q: (q["topic_id"], q["kind"], q["qid"]))


# Clearly off-syllabus questions (written for this benchmark, not by learners): the right behaviour is to decline.
OFF_TOPIC = [
    ("life_processes", "Explain Newton's third law of motion with an example."),
    ("life_processes", "Who won the cricket World Cup in 2011?"),
    ("control_and_coordination", "How do I solve a quadratic equation?"),
    ("control_and_coordination", "Why is the sky blue?"),
    ("how_do_organisms_reproduce", "What is the capital city of Australia?"),
    ("how_do_organisms_reproduce", "How does a refrigerator keep food cold?"),
    ("heredity", "What is Ohm's law?"),
    ("heredity", "How do I write a formal leave letter to my class teacher?"),
    ("our_environment", "What is the chemical formula of baking soda?"),
    ("our_environment", "How many players are there in a football team?"),
]


def load_doubts() -> list[dict[str, Any]]:
    """Learner-style doubts (evaluation/tagging/questions.csv, in scope) plus the off-topic set; no answer keys."""
    import csv
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "tagging" / "questions.csv"
    with open(path, encoding="utf-8-sig") as f:
        doubts = [{"qid": r["id"], "topic_id": r["topic_id"], "kind": "doubt", "query": r["question"]}
                  for r in csv.DictReader(f)]
    doubts += [{"qid": f"off{i + 1:02d}", "topic_id": t, "kind": "off_topic", "query": text}
               for i, (t, text) in enumerate(OFF_TOPIC)]
    return [{**d, "concept_id": None, "difficulty": None, "question": d["query"], "options": [],
             "answer_index": None} for d in doubts]


def _item(topic_id: str, kind: str, qid: str, it: dict[str, Any]) -> dict[str, Any]:
    return {"qid": qid, "topic_id": topic_id, "kind": kind, "concept_id": it.get("concept_id"),
            "difficulty": it.get("difficulty"), "question": it["question"], "options": list(it["options"]),
            "answer_index": int(it["answer_index"])}


def mcq_query(q: dict[str, Any]) -> str:
    options = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(q["options"]))
    return (f"{q['question']}\n{options}\nWhich option is correct? Start your answer with the option letter, "
            f"then explain briefly.")


_LEAD = re.compile(r"^\W*(?:answer\s*(?:is)?\s*[:\-]?\s*)?(?:option\s*)?\(?([A-D])\)?(?:[\s).:,\-]|$)", re.I)
_ANYWHERE = re.compile(r"(?:answer|correct (?:option|answer))\s*(?:is|:)?\s*(?:option\s*)?\(?\**([A-D])\b", re.I)


def parse_choice(text: str, options: list[str]) -> int | None:
    """The option the answer picks: a leading letter, "the answer is X", else the only option text it quotes."""
    text = (text or "").strip()
    for pattern in (_LEAD, _ANYWHERE):
        match = pattern.search(text.replace("**", ""))
        if match:
            return LETTERS.index(match.group(1).upper())
    quoted = [i for i, o in enumerate(options) if o and o.lower() in text.lower()]
    return quoted[0] if len(quoted) == 1 else None


# ---- variants -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    name: str
    route: str  # local | cloud | escalate
    top_k: int
    max_retries: int
    retrieval: bool = True
    family: str = "policy"  # policy (P1-P5) | ablation (RAG/no-retrieval controls for CRAG)


VARIANTS: dict[str, Variant] = {
    **{p.policy_id: Variant(p.policy_id, p.route, p.top_k, p.max_retries) for p in POLICIES.values()},
    # CRAG ablation at P2/P4's depth: the same retrieval without the corrective retry, and no retrieval at all
    "RAG-local": Variant("RAG-local", "local", 5, 0, family="ablation"),
    "RAG-cloud": Variant("RAG-cloud", "cloud", 5, 0, family="ablation"),
    "LLM-local": Variant("LLM-local", "local", 5, 0, retrieval=False, family="ablation"),
    "LLM-cloud": Variant("LLM-cloud", "cloud", 5, 0, retrieval=False, family="ablation"),
}


# ---- resource sampling ----------------------------------------------------------------------------------------

class ResourceSampler:
    """CPU/memory (psutil) and GPU utilisation/power (NVML) every `period_s` while a question runs."""

    def __init__(self, period_s: float = 0.2) -> None:
        self.period_s = period_s
        self._gpu = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # no NVIDIA GPU / driver: GPU columns stay empty
            self._nvml = None
        self._samples: list[tuple[float, float, float, float | None, float | None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ResourceSampler":
        self._samples.clear()
        self._stop.clear()
        psutil.cpu_percent(None)  # prime: the next call reports usage since now
        self._net0 = psutil.net_io_counters()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.period_s):
            gpu_util = gpu_w = None
            if self._gpu is not None:
                try:
                    gpu_util = self._nvml.nvmlDeviceGetUtilizationRates(self._gpu).gpu / 100
                    gpu_w = self._nvml.nvmlDeviceGetPowerUsage(self._gpu) / 1000
                except Exception:
                    pass
            self._samples.append((time.perf_counter(), psutil.cpu_percent(None) / 100,
                                  psutil.virtual_memory().percent / 100, gpu_util, gpu_w))

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self.elapsed_s = time.perf_counter() - self._t0
        net1 = psutil.net_io_counters()
        self.net_bytes = (net1.bytes_sent - self._net0.bytes_sent) + (net1.bytes_recv - self._net0.bytes_recv)

    def summary(self) -> dict[str, float | None]:
        s = self._samples
        mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
        cpu, mem = [x[1] for x in s], [x[2] for x in s]
        gpu_util = [x[3] for x in s if x[3] is not None]
        gpu_w = [x[4] for x in s if x[4] is not None]
        cpu_mean = mean(cpu)
        # CPU energy from the app's linear power model (an estimate); GPU energy measured by NVML (board power)
        cpu_wh = (None if cpu_mean is None else (settings.energy_idle_power_w + cpu_mean * (
            settings.energy_max_power_w - settings.energy_idle_power_w)) * self.elapsed_s / 3600)
        gpu_wh = mean(gpu_w) * self.elapsed_s / 3600 if gpu_w else None
        return {"cpu_mean": cpu_mean, "mem_max": max(mem) if mem else None, "gpu_util_mean": mean(gpu_util),
                "gpu_power_w": mean(gpu_w), "energy_cpu_wh_est": cpu_wh, "energy_gpu_wh": gpu_wh,
                "net_bytes": self.net_bytes, "samples": len(s)}


def current_network() -> str:
    """The Wi-Fi network (SSID) the machine is on, or "wired/unknown": every row records where it was measured."""
    import subprocess

    try:
        out = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True, timeout=5,
                             encoding="utf-8", errors="replace").stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    match = re.search(r"^\s*SSID\s*:\s*(.+)$", out, re.M)
    return match.group(1).strip() if match else "wired/unknown"


# ---- one answer -----------------------------------------------------------------------------------------------

class RateLimited(RuntimeError):
    """The cloud API refused the call for quota/rate reasons: the measurement must be retried, not recorded."""


@dataclass
class CallLog:
    calls: list[dict[str, Any]] = field(default_factory=list)


def _instrument(services: Any, log: CallLog) -> None:
    """Record every LLM call (router + P5 verifier): provider, model, latency, tokens, sizes."""
    if getattr(services, "_bench_instrumented", False):
        return

    def wrap(client: Any, kind: str) -> None:
        inner = client.generate

        def generate(request: Any) -> Any:
            start = time.perf_counter()
            try:
                result = inner(request)
            except Exception as exc:
                log.calls.append({"kind": kind, "provider": getattr(client, "provider", "?"), "ok": False,
                                  "ms": (time.perf_counter() - start) * 1000, "error": str(exc)[:300]})
                if re.search(r"\b429\b|rate limit|tokens per", str(exc), re.I):
                    raise RateLimited(str(exc)) from exc
                raise
            log.calls.append({"kind": kind, "provider": result.provider, "model": result.model, "ok": True,
                              "ms": result.latency_ms, "completion_tokens": result.completion_tokens,
                              "prompt_chars": sum(len(m["content"]) for m in request.messages()),
                              "answer_chars": len(result.text)})
            return result

        client.generate = generate

    for name, client in services.router.clients.items():
        wrap(client, "answer")
    if services.verifier is not None:
        wrap(services.verifier, "check")
    services._bench_instrumented = True


def answer(q: dict[str, Any], variant: Variant, *, sampler: ResourceSampler | None = None,
           judge: bool = False, scaffold: dict[str, str] | None = None) -> dict[str, Any]:
    """Run the question through the QA nodes with this variant; returns one result row (raises RateLimited).

    scaffold: the coach's answer shape ({style, support}) to apply, as coach_mode=on would (None: plain answer).
    """
    services = get_services()
    log = getattr(services, "_bench_log", None) or CallLog()
    services._bench_log = log
    _instrument(services, log)
    log.calls.clear()
    services.router._health.clear()  # a failure in an earlier question must not reroute this one

    query = q.get("query") or mcq_query(q)  # learner doubts carry their own text and no options
    state: dict[str, Any] = {
        "session_id": "benchmark", "student_id": "benchmark", "topic_id": q["topic_id"], "resource_id": q["topic_id"],
        "runtime_config": {"top_k": variant.top_k, "threshold": settings.crag_retrieval_threshold,
                           "max_retries": variant.max_retries, "route": variant.route},
        "query": query, "retrieval_query": query, "question_started_at": time.time(), "q_errors": [],
        "chunks": [], "retrieval_score": 0.0, "crag_retries": 0, "model_used": "", "response": "",
        "escalation": {}, "qa_log": [], "competency": "intermediate", "profile": {"language": "english"},
        "coach_mode": "off", "question_concept": {}, "video": None,
    }
    timings = {"retrieve_ms": 0.0, "route_ms": 0.0, "generate_node_ms": 0.0}
    sampler = sampler or ResourceSampler()
    if not variant.retrieval:
        return _closed_book(q, variant, query, sampler, log)
    with sampler:
        started = time.perf_counter()
        if variant.retrieval:
            while True:
                t = time.perf_counter()
                state.update(qa.crag_retrieve(state))
                timings["retrieve_ms"] += (time.perf_counter() - t) * 1000
                if crag_ok(state) == "llm_router":
                    break
                state.update(qa.corrective_retrieve(state))
        t = time.perf_counter()
        state.update(qa.llm_router(state))
        timings["route_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        if scaffold:
            import agent.nodes.coaching as coaching

            original, coaching.scaffold_for = coaching.scaffold_for, lambda _state: dict(scaffold)
            try:
                out = qa.generate_response(state)
            finally:
                coaching.scaffold_for = original
        else:
            out = qa.generate_response(state)
        timings["generate_node_ms"] = (time.perf_counter() - t) * 1000
        total_ms = (time.perf_counter() - started) * 1000
    entry = out["qa_log"][0]
    row = _row(q, variant, entry, out["response"], state, timings, total_ms, log, sampler)
    row["scaffold"] = "+".join(v for k, v in (scaffold or {}).items() if k in ("style", "support")) or "none"
    if judge:  # faithfulness of the final answer to the retrieved chunks (P5's checker, outside the timed part)
        from agent.escalation import check_answer

        check = check_answer(services.verifier, state["query"], [c["text"] for c in state["chunks"]], out["response"])
        row.update({"faithfulness": check.score, "faithfulness_issue": check.issue, "judge_error": check.error,
                    "declined": int(bool(DECLINED.search(out["response"])))})
    return row


DECLINED = re.compile(r"(does not|doesn't|did not|isn't|is not|not) (cover|covered|include|mention|contain)"
                      r"|not (in|part of) (the|this) lesson|outside (the|this) lesson", re.I)


CLOSED_BOOK_SYSTEM = ("You are a patient teaching assistant for school learners. Answer from your own knowledge. "
                      "Keep the answer under 120 words.")


def _closed_book(q: dict[str, Any], variant: Variant, query: str, sampler: ResourceSampler,
                 log: CallLog) -> dict[str, Any]:
    """Control without retrieval: the same model, a neutral prompt (the app's prompt declines without context)."""
    from services.llm_clients import GenerationRequest

    request = GenerationRequest(system=CLOSED_BOOK_SYSTEM, prompt=f"Question: {query}\nAnswer:",
                                max_tokens=settings.llm_max_tokens, temperature=settings.llm_temperature,
                                operation="generate-answer")
    router = get_services().router
    with sampler:
        started = time.perf_counter()
        error = None
        try:
            result = router.generate(variant.route, request)
            text, model_used = result.text, f"{result.provider}:{result.model}"
        except RateLimited:
            raise
        except Exception as exc:  # recorded as a failed answer, like the app's fallback chain would show
            text, model_used, error = "", f"{variant.route}:failed", str(exc)[:300]
        total_ms = (time.perf_counter() - started) * 1000
    entry = {"model_used": model_used, "n_chunks": 0, "escalated": False, "escalation_stage": None,
             "check_score": None, "check_ms": None, "error": error}
    state = {"retrieval_score": 0.0, "crag_retries": 0}
    timings = {"retrieve_ms": 0.0, "route_ms": 0.0, "generate_node_ms": total_ms}
    return _row(q, variant, entry, text, state, timings, total_ms, log, sampler)


def _row(q: dict[str, Any], variant: Variant, entry: dict[str, Any], response: str, state: dict[str, Any],
         timings: dict[str, float], total_ms: float, log: CallLog, sampler: ResourceSampler) -> dict[str, Any]:
    choice = parse_choice(response, q["options"]) if q.get("options") else None
    out = {"response": response}
    stamp = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "network": current_network()}
    answers = [c for c in log.calls if c["kind"] == "answer"]
    checks = [c for c in log.calls if c["kind"] == "check"]
    cloud = [c for c in log.calls if c.get("provider") == "cloud"]
    return {
        **stamp,
        "qid": q["qid"], "topic_id": q["topic_id"], "kind": q["kind"], "concept_id": q["concept_id"],
        "difficulty": q["difficulty"], "variant": variant.name, "family": variant.family, "route": variant.route,
        "top_k": variant.top_k, "max_retries": variant.max_retries, "retrieval": variant.retrieval,
        "model_used": entry["model_used"], "provider": entry["model_used"].split(":", 1)[0],
        "retrieval_score": round(state["retrieval_score"], 4), "crag_retries": state["crag_retries"],
        "n_chunks": entry["n_chunks"], "escalated": entry["escalated"], "escalation_stage": entry["escalation_stage"],
        "check_score": entry["check_score"], "check_ms": entry["check_ms"],
        "choice": None if choice is None else LETTERS[choice],
        "key": LETTERS[q["answer_index"]] if q.get("answer_index") is not None else None,
        "correct": None if choice is None else int(choice == q["answer_index"]), "parsed": int(choice is not None),
        "total_ms": round(total_ms, 1), **{k: round(v, 1) for k, v in timings.items()},
        "llm_ms": round(sum(c["ms"] for c in answers if c["ok"]), 1),
        "n_llm_calls": len(answers), "n_cloud_calls": len(cloud), "n_check_calls": len(checks),
        "completion_tokens": sum(c.get("completion_tokens") or 0 for c in answers),
        "prompt_chars": sum(c.get("prompt_chars") or 0 for c in log.calls),
        # per route, for the network model: what crossed the network and how long the cloud calls took
        "cloud_ms": round(sum(c["ms"] for c in cloud), 1),
        "local_ms": round(sum(c["ms"] for c in log.calls if c.get("provider") == "local"), 1),
        "cloud_prompt_chars": sum(c.get("prompt_chars") or 0 for c in cloud),
        "cloud_answer_chars": sum(c.get("answer_chars") or 0 for c in cloud),
        "error": entry["error"], **sampler.summary(), "response": out["response"],
    }
