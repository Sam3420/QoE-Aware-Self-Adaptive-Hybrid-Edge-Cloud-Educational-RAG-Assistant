# Policy P5: confidence-aware local → cloud escalation

P5 (`local-escalate`) is a fifth runtime policy. It answers with the **local SLM (Ollama, gemma3:1b)** by default and uses
the **cloud LLM (Groq, gpt-oss-120b)** only for questions where the local answer is likely to be weak or is shown to be
unsupported. It aims for most of the correctness of the cloud model at close to the cost and latency of the local one.

P1–P4 are unchanged, so they remain clean baselines. The adaptive policy engine treats P5 as one more action and learns from
experience **when** the cascade pays off.

## The policy catalogue

| Policy | Route | top_k | CRAG retries | What it does |
|---|---|---|---|---|
| P1 local-light (default) | local | 3 | 0 | Local SLM only |
| P2 local-thorough | local | 5 | 1 | Local SLM only |
| P3 cloud-light | cloud | 3 | 0 | Cloud LLM only |
| P4 cloud-thorough | cloud | 5 | 1 | Cloud LLM only |
| **P5 local-escalate** | **escalate** | 5 | 1 | Local SLM first; cloud LLM when needed (below) |

`escalate` is a third value of the existing `route` column, so the `policies` table needs no schema change. The row is
added to existing databases on the next start, because `seed_policies` inserts missing policies.

## How one question is answered under P5

```
learner question
   │
   ▼
CRAG retrieval + corrective retries (as for every policy)
   │
   ▼
llm_router ── retrieval score in the borderline band [0.55, 0.65)? ──yes──► cloud LLM answers   (escalated "before")
   │ no                                                                         (not checked)
   ▼
local SLM answers, streamed to the learner immediately
   │
   ▼
grounding check (Groq gpt-oss-20b): share of the answer's claims supported by the retrieved chunks
   │
   ├─ score ≥ 0.75 → the local answer stands
   └─ score < 0.75 → cloud LLM answers again; the streamed draft is replaced  (escalated "after")
```

### Trigger 1: before answering (borderline retrieval)
- **Where:** `agent/escalation.reason_to_escalate_before`, called by the `llm_router` node.
- **Rule:** escalate when the retrieval score, after CRAG's corrective retries, is at least the floor (0.55) but below the
  CRAG threshold (0.65). The evidence is in the video but weakly matched, and the stronger model makes better use of it.
- **Below the floor** the question is off-topic for the video (nomic: off-topic 0.47–0.55, on-topic 0.71–0.82). The cloud
  can't add content the video doesn't contain, so the local model answers "not covered" at no cloud cost.
- **Why the floor exists:** in the first live test, escalating clearly off-topic questions cost about 1 s and one cloud
  call each, and only produced the same "not covered" reply.

### Trigger 2: after a local answer (grounding check)
- **Where:** `agent/escalation.check_answer`, called by `generate_response` through `_check_and_escalate`.
- **Check model:** Groq `openai/gpt-oss-20b`, a small, fast model (temperature 0). It splits the answer into claims and
  returns `{"score": supported/all, "issue": "..."}`.
- **Outcome:** below 0.75 the cloud LLM answers again with the same prompt and chunks. The streamed draft is cleared, using
  the streaming hook's `sink(None)`, and the cloud answer is shown.
- **The check never blocks an answer.** If the check model fails, replies with invalid JSON, or no Groq key is set, the
  local answer stands and the reason is logged.

## How it fits the design document

| Design element | How P5 implements it |
|---|---|
| **Phase 3: Hybrid router.** Inputs are retrieval confidence, query complexity and runtime policy. | P5 makes **retrieval confidence** an actual router input (trigger 1). Query complexity is still not implemented. |
| **Phase 4 / Algorithm 5: the adaptive policy picks a runtime configuration.** | P5 is one more action in the candidate set. The FAISS kNN estimates its utility like any other policy. Adaptation still changes only predefined runtime settings, never code. |
| **Hybrid edge–cloud inference: balance efficiency, quality, cost and latency.** | A cascade: edge model first, cloud model only for weak or unsupported answers. |
| **AI quality (AQ): RAG quality, LLM response quality.** | Wrong local answers are replaced before the learner keeps them, which raises answer quality (LQ). Samridhi's Langfuse judge measures LQ on the **final** answer of each question (`telemetry/answer_quality.py` keeps the latest `generate-answer` per trace). |
| **NQ: pipeline latency (PL), availability.** | PL includes the check and any escalation, so P5 pays its cost in QoE. |
| **SRE: network efficiency.** | Each check and escalation is cloud traffic, counted in the network bytes SRE measures. |
| **Algorithm 1: reward R = µ1·ΔQoE + µ2·ΔSRE.** | P5's reward reflects both sides: better answers (QoE) against latency and network cost (QoE and SRE). Whether the trade-off pays off is what the experience memory learns. |
| **Continuous monitoring and observability (Phase 6).** | Every decision is logged per question and traced in Langfuse (below). |

## What is recorded

**Per question, in `qa_log`** (the checkpoint and the session summary):

| Field | Meaning |
|---|---|
| `preferred_route` | `escalate` for P5 |
| `model_used` | Which model produced the final answer |
| `escalated`, `escalation_stage` | `before` (borderline retrieval) or `after` (failed check) |
| `escalation_reason` | e.g. `check score 0.00 < 0.75: Australia won the cricket World Cup in 2011` |
| `check_supported`, `check_score`, `check_ms` | Grounding-check verdict, score and time; `None` if not run |
| `draft_model` | The local model whose answer was replaced |

**In Langfuse,** inside each `answer-question` trace:
- `route-llm` shows the retrieval score and the escalation decision;
- `verify-answer` is a generation for the check;
- the replacement appears as a second `generate-answer` generation.

**In the app:**
- the chat shows "Escalated to the cloud model … because …", or the passed check with its score;
- *Technical details* lists the escalation stage and the check;
- *System capabilities* lists P5.

## Settings (`config.py`, override in `.env`)

| Setting | Default | Purpose |
|---|---|---|
| `ESCALATION_CHECK_ENABLED` | `true` | Turn the after-answer check on or off |
| `ESCALATION_CHECK_MODEL` | `openai/gpt-oss-20b` | Check model on Groq |
| `ESCALATION_CHECK_MAX_TOKENS` | `400` | Includes the model's reasoning tokens |
| `ESCALATION_MIN_CHECK_SCORE` | `0.75` | Below this share of supported claims, the answer is replaced |
| `ESCALATION_MIN_RETRIEVAL_SCORE` | `0.55` | Floor of the "before" band; the top of the band is the CRAG threshold (0.65) |

These need `GROQ_API_KEY` (or `GROK_API_KEY`). Without it, P5 behaves like P2 (local only, no check).

## Live test (17 Oct 2026)

**Setup:**
- Real Ollama gemma3:1b, Groq gpt-oss-120b (answers) and gpt-oss-20b (check), nomic embeddings.
- Video: "Life process class 10 Animated video" (human transport system, 9 chunks).
- P5 was made the default in a scratch database.
- **Five questions only, so this is illustrative, not a result.**

| Question | Retrieval | Outcome | Check | Total time |
|---|---|---|---|---|
| Why do arteries have thick elastic walls? | 0.71 | **Replaced** by cloud ("great pressure" explanation) | 0.50 in 1.0 s | 6.6 s (local model was loading) |
| What is double circulation and why do birds and mammals need it? | 0.73 | Local kept | 1.00 in 1.8 s | 2.6 s |
| How do platelets stop bleeding after a cut? | 0.72 | Local kept | 1.00 in 1.0 s | 1.4 s |
| What is lymph and where does it come from? | 0.72 | Local kept | 1.00 in 0.9 s | 1.4 s |
| Who won the cricket world cup in 2011? | 0.46 (below floor) | gemma hallucinated "Australia" (India won) → **replaced** with "not covered" | 0.00 in 0.8 s | 2.6 s |

**Observations:**
- **3 of 5** answers stayed local. Both wrong local answers were caught.
- The two triggers complement each other. The floor kept the off-topic question local, and the check then caught the
  hallucination that the local model produced.
- **The check costs about 1 s per local answer**, more than first estimated. The learner already sees the streamed local
  answer during the check, but PL includes it.

## Limitations

1. **Check latency (~0.8–1.8 s).** A faster check model (Groq `qwen/qwen3.8-27b` answered in ~0.55 s in our tests) or
   checking only some answers would reduce it. Measure before changing.
2. **The check measures support by the video, not truth.** A claim that is correct but absent from the chunks counts as
   unsupported, which leads to unnecessary escalations.
3. **The check model is itself an LLM.** Calibrate it the same way as the LQ judge: compare with about 30 hand-labelled
   answers and report the agreement.
4. **Experience needed.** With five policies the kNN selector needs more sessions per policy before it trusts P5.
   Cold start (3 experiences) and ε = 0.2 exploration still apply. For experiments, also run P5 as a fixed policy.
5. **Free-tier quota.** Every check and escalation uses Groq requests; check your limits before experiment batches.

## Metrics to report in experiments

- **Escalation rate:** before, after, and total, per condition.
- **Correction rate:** after-escalations ÷ local answers checked.
- **Latency cost:** PL of P5 vs P2 (same retrieval, no check) and vs P4 (cloud only).
- **LQ** (Langfuse judge) for P2, P4 and P5. Expectation: P5 close to P4, P2 lowest.
- **SRE / network bytes** for P2, P4 and P5. Expectation: P5 between P2 and P4.
- **How often the adaptive selector picks P5,** by context (RTT, CPU, competency).

## Files

| File | Change |
|---|---|
| `agent/escalation.py` | New: the borderline rule and the grounding check |
| `agent/nodes/qa.py` | `llm_router` escalates before; `generate_response` checks and escalates after; `qa_log` fields |
| `agent/policies.py`, `app/core/config.py` (seed) | P5 and the `escalate` route mode |
| `agent/schemas.py`, `agent/state.py` | `escalate` route, per-question escalation fields |
| `agent/deps.py` | `make_verifier()`: the check model on Groq |
| `config.py` | `ESCALATION_*` settings |
| `telemetry/answer_quality.py` | LQ scores the final answer of each question |
| `agent/node_tracing.py`, `frontend/…` | Tracing and UI display |
| `tests/test_escalation_policy.py`, `tests/test_answer_quality.py` | Tests for P5 and final-answer LQ |
