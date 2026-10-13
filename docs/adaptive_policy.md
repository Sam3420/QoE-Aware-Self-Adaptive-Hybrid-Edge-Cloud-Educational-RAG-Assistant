# QoE-aware adaptive policy learning

A closed loop inside the single LangGraph agent. For each learning session it:
1. measures the context
2. retrieves similar past sessions with FAISS
3. estimates the utility of each predefined policy
4. applies the best one
5. measures the session
6. computes QoE, SRE, utility and the reward
7. stores the transition for future decisions

**This is not reinforcement learning.** No policy or value function is trained. Past transitions are only looked up.

## Where it lives

| Algorithm | Module | Notes |
|---|---|---|
| Normalisation, weights, missing-data policy | `metrics/normalization.py`, `metrics/weights.py` | pure |
| QoE | `metrics/qoe.py` | pure |
| SRE | `metrics/sre.py` | pure |
| Utility | `metrics/utility.py` | pure |
| Reward | `metrics/reward.py` | pure; persistence is in the store |
| Entropy Weight Method | `metrics/ewm.py` (algorithm), `agent/weight_estimation.py` (data + persistence) | |
| Feedback sentiment (estimate) | `metrics/sentiment.py` | lexicon |
| Session → metric mapping | `agent/session_metrics.py` | the only place that knows where each metric comes from |
| Experience store (DB + FAISS) | `agent/experience_store.py`, `services/experience_index.py` | |
| Policy selection | `agent/policy_engine.py` | |
| Loop integration | `agent/nodes/policy.py` (`select_policy`), `agent/nodes/evaluation.py` (`evaluate_session`, `store_experience`), `agent/nodes/context.py` | |
| Tracing and logging | `agent/observability.py` | Langfuse optional |
| Tables | `app/db/models/adaptive_learning.py`, `app/db/repositories/adaptive_learning.py` | additive |
| CLI | `scripts/adaptive_policy.py` | |

## The loop and its transitions

One transition is **one learning session**:

| Symbol | Meaning |
|---|---|
| `s` | The learner's previous measured state: the `s_next` stored with their last transition. On a first session, `s` is a baseline state and `QoE_old`/`SRE_old` = `BASELINE_QOE`/`BASELINE_SRE` (flagged `baseline_used`). |
| `a` | The predefined policy P1–P4 executed this session: `top_k`, CRAG `max_retries`, local/cloud `route` (the agent applies the CRAG threshold from config). These are the agent's real runtime adaptations. |
| `s_next` | This session's measured state. |

The graph runs these steps:

1. `collect_context` measures the context (RTT probe, CPU and memory, competency, last QoE) and the network counter.
2. `select_policy`:
   1. loads `s`
   2. queries FAISS
   3. estimates Û(s, a) for every candidate
   4. picks a*
   5. records the estimates (predictions)
3. CRAG retrieval and the LLM router execute a* (`top_k`, `max_retries`, `route`).
4. `evaluate_session`:
   1. samples resources again
   2. builds `s_next` from the session's measurements
   3. scores `s` and `s_next`
   4. computes R

   If no question was answered, the policy never ran: status `not_executed`, no reward.
5. `store_experience` persists (s, a, R, s_next) **only** for completed transitions. The `experiences` row (Person B), `experience_transitions` and `experience_vectors` are written in one transaction, then the vector is added to FAISS.
6. Scores, decisions and rewards are exposed in the agent state (`evaluation`, `candidate_utilities`, `similar_exps`, `decision_reason`), as structured JSON logs (logger `adaptive`), in Langfuse spans, and in the Streamlit summary, Monitoring and Adaptation pages.

## Equations

All inputs are normalised to [0, 1] with higher = better. Each weight group must have exactly its keys, be non-negative, and sum to 1; this is validated at startup.

### QoE (`metrics/qoe.py`)
```
EQ  = α1·LG + α2·PT + α3·RA + α4·LS
AQ  = β1·RQ + β2·LQ + β3·SQ + β4·TQ
NQ  = γ1·PL + γ2·NR + γ3·RU + γ4·AV
UX  = δ1·SR + δ2·FS + δ3·CR
QoE = wEQ·EQ + wAQ·AQ + wNQ·NQ + wUX·UX
```

| Metric | Measured as (`agent/session_metrics.py`) | Normalisation |
|---|---|---|
| LG learning gain | Hake gain (post − pre)/(1 − pre) | gain ∈ [−1, 1] → (g + 1)/2 |
| PT post-test | post-test score | fraction |
| RA recommendation acceptance | accepted / shown video recommendations, same topic, since the previous session | fraction |
| LS learner satisfaction | mean star rating of those recommended videos | 1–5 → 0–1 |
| RQ RAG quality | mean CRAG retrieval score per question | benefit, `QOE_RETRIEVAL_FLOOR` (0.40) → 0, `QOE_RETRIEVAL_CEIL` (0.85) → 1 |
| LQ LLM response quality | **unavailable**: no evaluator in the project | — |
| SQ, TQ speech quality | **unavailable**: STT/TTS not implemented | — |
| PL latency quality | mean end-to-end answer latency | **cost**: ≤ `QOE_LATENCY_GOOD_MS` (1500) → 1, ≥ `QOE_LATENCY_BAD_MS` (15000) → 0 |
| NR network reliability | successful / attempted network operations: answers without a route or generation failure, plus the start and end RTT probes | fraction |
| RU utilisation quality | peak of CPU and memory at start and end | **health band**: ≤ `QOE_UTILIZATION_TARGET` (0.70) → 1, ≥ `QOE_UTILIZATION_SATURATION` (0.95) → 0. Low use is not penalised |
| AV availability | answers from an LLM route (not the extractive fallback) / questions | fraction |
| SR star rating | session rating | 1–5 → 0–1 |
| FS sentiment | **estimate**: lexicon sentiment of the comment (`SENTIMENT_LEXICON_ENABLED`) | [−1, 1] → [0, 1] |
| CR completion | completed learning steps: pre-test (or already taken), ≥ 1 question, post-test | fraction |

### SRE (`metrics/sre.py`)
```
X_eff = 1 − usage_X / max_X  (clipped to [0, 1])
SRE   = η1·CPU_eff + η2·MEM_eff + η3·EN_eff + η4·NE_eff
```

| Resource | Usage | Budget |
|---|---|---|
| CPU | mean system utilisation at start and end (psutil) | `SRE_CPU_MAX` (1.0) |
| Memory | same | `SRE_MEMORY_MAX` (1.0) |
| Energy | **estimate**: (P_idle + (P_max − P_idle)·u_cpu) × busy time (sum of answer latencies). GPU and cloud energy are not included | `SRE_ENERGY_MAX_WH` (2 Wh/session) |
| Network | system-wide bytes (psutil) between context collection and evaluation | `SRE_BANDWIDTH_MAX_BYTES` (50 MB/session) |

SRE alone would reward doing less. It is never used alone: utility and reward always combine it with QoE, which falls when the service degrades.

### Utility and reward
```
U(s, a) = λ1·QoE + λ2·SRE              (observed state)
R       = μ1·(QoE_new − QoE_old) + μ2·(SRE_new − SRE_old)
```
Improvement gives R > 0, regression gives R < 0, and no change gives R = 0. `QoE_old`/`SRE_old` are **recomputed** from the metrics stored with `s`, using the current weights, so a reward is deterministic for given inputs and configuration. The weights version is stored with every transition.

### Missing data (`METRIC_MISSING_POLICY`)
The same policy applies inside each group and across groups:

| Policy | Behaviour |
|---|---|
| `renormalize` (default) | Σ_available w·x / Σ_available w. Every result reports `coverage` (the share of weight backed by data), the missing metrics and the missing groups. A group with no data is **unavailable, never 0**. |
| `neutral` | Missing values count as 0.5. |
| `strict` | Raise `MissingMetricError`. |

If `s_next` has no measurable QoE or SRE at all, the transition is `incomplete` and is not stored.

## Policy selection (`agent/policy_engine.py`)

- **State embedding:** the context `{rtt_ms, cpu, mem, competency_num, last_qoe}` becomes a [0, 1]^5 vector. RTT is divided by `RTT_MAX_MS` and clipped, and a missing `last_qoe` becomes `DEFAULT_LAST_QOE`.
  The same `StateEmbedder` is used for indexing and querying, and its version string includes the normalisation parameters. If they change, every transition is re-embedded from its stored state.
- **Retrieval:** FAISS `IndexIDMap2(IndexFlatL2)` (exact) returns the top `KNN_K` states. Their ids map to experience ids in `experience_vectors`, and the full records come from the database.
- **Estimate per candidate a:** only neighbours that **executed a** count. With w_i = exp(−d_i² / 2σ²) (σ = `POLICY_KERNEL_SIGMA`, 0.25):
  ```
  Û(s, a) = Σ w_i·U_i / Σ w_i,   U_i = λ1·QoE_after_i + λ2·SRE_after_i (recomputed with current weights)
  ```
  Set `POLICY_ESTIMATE_TARGET=reward` to use R_i instead.
  An action has **no estimate** unless it has ≥ `POLICY_MIN_SUPPORT` neighbours and Σw ≥ `POLICY_MIN_WEIGHT_SUM` (0.1, roughly one neighbour within 2σ). Evidence from very different states is not trusted, however good its value.
- **Choice rules:**
  - **Cold start** (< `MIN_EXPERIENCES` stored transitions): use the default policy (P1).
  - **No candidate has an estimate:** fall back to the default.
  - **Otherwise, ε-greedy** with a seeded RNG (`EPSILON`, `POLICY_RNG_SEED`). With probability ε, pick a random *other* candidate; this is how unseen policies get tried. Otherwise pick the highest Û, with ties going to the default.
- **Prediction is not observation.** Û of the chosen policy is stored as `predicted_utility`, and the measured `utility_after` is stored separately. The engine never uses post-adaptation measurements before they exist.

## Entropy Weight Method (`metrics/ewm.py`, `agent/weight_estimation.py`)

For one group's observation matrix (n rows × m metrics):
```
x' = min–max(x) per column,  p_ij = x'_ij / Σ_i x'_ij,
e_j = −(1/ln n) Σ_i p_ij ln p_ij,  d_j = 1 − e_j,  w_j = d_j / Σ d_j
```

- **Incomplete data:** rows with missing values are dropped.
- **Constant columns** get d = 0.
- **Fallback to the baseline weights** (with the reason stored) when there are fewer than `EWM_MIN_OBSERVATIONS` complete rows, every column is constant, or the result is not finite.
- **One group at a time.** Each group is estimated only from its own metrics in the stored `s_next`: the eq/aq/nq/ux metrics, the QoE components, the SRE efficiencies, (QoE, SRE) after for utility, and (ΔQoE, ΔSRE) for reward.
- **Inspectable.** Every result is stored in `weight_sets` with its data window, entropy, divergence and a baseline-vs-derived comparison.
- **Explicit opt-in.** A result is used only if it is **activated** *and* `METRIC_WEIGHTS_SOURCE=ewm`. A fallback is never activated.

**EWM weights measure how much a metric varies in the data, not how important it is.** Treat them as a data-driven alternative to compare, not as ground truth.

## Database tables (additive; created automatically by `create_all`)

| Table | Content |
|---|---|
| `experiences` (Person B) | context state, policy, QoE/SRE before and after, reward, timestamp |
| `experience_transitions` | same id: action (+ config), s, s_next (metrics), component scores, utility before/after, **predicted utility**, baseline flag, weights version. Unique per `session_id`, which makes writes idempotent |
| `experience_vectors` | `faiss_id` ↔ `experience_id`, the embedded vector, the embedding version (FAISS can always be rebuilt from it) |
| `weight_sets` | EWM or fallback results per group, data window, active flag |

FAISS files are stored in `data/experience_index/` and are never the source of truth. `sync_index()` rebuilds the index whenever its ids differ from `experience_vectors`.
Experiences stored before this module existed have no transition record, so they are not indexed or used.

## Configuration (`.env` / `config.py`)

| Key | Default |
|---|---|
| `METRIC_WEIGHTS` (JSON of groups) | qoe {EQ .35, AQ .25, NQ .20, UX .20}, eq {LG .35, PT .30, RA .15, LS .20}, aq {RQ .40, LQ .30, SQ .15, TQ .15}, nq {PL .35, NR .25, RU .15, AV .25}, ux {SR .45, FS .20, CR .35}, sre {CPU .35, MEM .30, EN .20, NE .15}, utility {QOE .7, SRE .3}, reward {QOE .7, SRE .3} |
| `METRIC_WEIGHTS_SOURCE` | `fixed` (or `ewm`) |
| `METRIC_MISSING_POLICY` | `renormalize` |
| `QOE_*`, `SRE_*`, `ENERGY_*` bounds | see the tables above |
| `BASELINE_QOE`, `BASELINE_SRE` | 0.5, 0.5 |
| `KNN_K`, `MIN_EXPERIENCES`, `EPSILON`, `POLICY_RNG_SEED` | 5, 3, 0.2, none |
| `POLICY_KERNEL_SIGMA`, `POLICY_MIN_SUPPORT`, `POLICY_MIN_WEIGHT_SUM`, `POLICY_ESTIMATE_TARGET` | 0.25, 1, 0.1, `utility` |
| `EXPERIENCE_INDEX_DIR` | `data/experience_index` |
| `EWM_MIN_OBSERVATIONS` | 10 |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | unset = tracing off |

## Commands

```powershell
# install (Langfuse is optional)
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install langfuse            # only if you want tracing

# initialise storage and check it
.venv\Scripts\python scripts\adaptive_policy.py init
.venv\Scripts\python scripts\adaptive_policy.py status

# dry-run a decision for a context (nothing is executed or stored)
.venv\Scripts\python scripts\adaptive_policy.py select --rtt 300 --cpu 0.4 --mem 0.6 --competency 0.5

# entropy weights: estimate, compare with the baseline, optionally activate (and set METRIC_WEIGHTS_SOURCE=ewm)
.venv\Scripts\python scripts\adaptive_policy.py estimate-weights --all
.venv\Scripts\python scripts\adaptive_policy.py estimate-weights --group eq --activate

# run the loop
.venv\Scripts\python scripts\run_stub_session.py      # offline, synthetic, self-checking
.venv\Scripts\python scripts\run_session.py           # real services
.venv\Scripts\streamlit run frontend\streamlit_app.py

# tests
.venv\Scripts\python -m pytest -q
```

## Limitations
- **Measurement gaps.** LLM-response, STT and TTS quality have no data source, so AQ rests on RAG quality alone (coverage is reported). Sentiment and energy are labelled estimates. CPU, memory and bandwidth are system-wide and include other processes.
- **Session granularity.** Transitions are per session, so ΔQoE also reflects the topic and the learner's own progress, not only the policy. There is no causal attribution. Topic is not part of the state vector, matching Person B's `ExperienceState`.
- **Exploration.** Exploration is ε-greedy without uncertainty bonuses. With little data, estimates rest on a few neighbours (the support and weight sums are shown with every decision).
- **Video recommendations** (strategies R1–R3) still use the earlier kNN (`agent/policy_selection.py`) and a placeholder reward, because recommendation runs have no before/after state yet. Their acceptance and ratings do feed session QoE (RA, LS).
- **Index scale.** The FAISS index is exact (`IndexFlatL2`), which is right for thousands of transitions; switch to an IVF index beyond that.
