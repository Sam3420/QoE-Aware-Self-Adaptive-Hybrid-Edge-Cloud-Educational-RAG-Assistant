"""Policy P5 (route "escalate"): a confidence-aware local → cloud cascade for one question.

1. Before answering: if CRAG's retrieval confidence is borderline (at least ESCALATION_MIN_RETRIEVAL_SCORE but
   below the policy threshold, after the corrective retries), the evidence is present but weakly matched, so the
   question goes to the cloud LLM, which uses weak evidence better. Below the floor the video does not cover the
   question; the cloud cannot add the missing content, so the local model answers ("not covered").
2. After a local answer: a small, fast cloud model checks whether the answer's claims are supported by the
   retrieved chunks. If the share of supported claims is below ESCALATION_MIN_CHECK_SCORE, the cloud LLM answers
   again and replaces the local answer.

The other policies never escalate: P1/P2 stay local and P3/P4 stay cloud, so they remain clean baselines and the
adaptive policy engine learns when the cascade pays off (see docs/escalation_policy.md).
"""
import json
import re
import time
from typing import Any

from pydantic import BaseModel

from config import settings
from services.llm_clients import GenerationRequest, LLMError

CHECK_SYSTEM = ("You check a tutor's answer against lesson excerpts. Judge only whether the answer's claims are "
                "supported by the excerpts. Reply with one JSON object and nothing else.")


class CheckResult(BaseModel):
    """Grounding check of one answer. `supported` is None when the check could not run (the answer then stands)."""

    supported: bool | None = None
    score: float | None = None  # share of the answer's claims supported by the excerpts, 0-1
    issue: str = ""
    ms: float = 0.0
    error: str | None = None


def reason_to_escalate_before(retrieval_score: float, threshold: float,
                              floor: float | None = None) -> str | None:
    """Why P5 should skip the local model for this question, or None to answer locally."""
    floor = settings.escalation_min_retrieval_score if floor is None else floor
    if floor <= retrieval_score < threshold:
        return f"borderline retrieval confidence {retrieval_score:.2f} (between {floor:.2f} and threshold {threshold:.2f})"
    return None


def check_prompt(question: str, chunks: list[str], answer: str) -> str:
    excerpts = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(chunks, start=1)) or "(no excerpts)"
    return (
        f"Lesson excerpts:\n{excerpts}\n\nQuestion: {question}\n\nTutor's answer: {answer}\n\n"
        "Split the tutor's answer into its factual claims and decide for each whether the excerpts support it. "
        "Statements that the lesson does not cover something count as supported. Reply with JSON: "
        '{"score": <supported claims / all claims, 0 to 1>, "issue": "<the main unsupported or wrong claim, '
        'or empty>"}'
    )


def check_answer(verifier: Any, question: str, chunks: list[str], answer: str) -> CheckResult:
    """Ask the verifier model how well the excerpts support the answer (never raises)."""
    request = GenerationRequest(system=CHECK_SYSTEM, prompt=check_prompt(question, chunks, answer),
                                max_tokens=settings.escalation_check_max_tokens, temperature=0.0,
                                operation="verify-answer")
    start = time.perf_counter()
    try:
        result = verifier.generate(request)
        data = _json_object(result.text)
        score = min(max(float(data["score"]), 0.0), 1.0)
    except (LLMError, KeyError, TypeError, ValueError) as exc:
        return CheckResult(ms=(time.perf_counter() - start) * 1000, error=f"{type(exc).__name__}: {exc}"[:300])
    return CheckResult(supported=score >= settings.escalation_min_check_score, score=round(score, 3),
                       issue=str(data.get("issue") or "")[:300], ms=(time.perf_counter() - start) * 1000)


def _json_object(text: str) -> dict[str, Any]:
    """The first JSON object in the reply (models sometimes wrap it in a code fence)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        raise ValueError("no JSON object in the check reply")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("check reply is not a JSON object")
    return data
