# Benchmarks for the Results chapter

This folder produces every number, table and figure for the **Results and Discussion** chapter. No students or
simulated learners are involved. Everything is measured on the system itself, using two kinds of input:

- **fixed question sets:**
  - the 150 keyed multiple-choice items from the 5 content packages;
  - 30 learner-style doubts (`evaluation/tagging/questions.csv`);
  - 10 off-topic questions;
- **network conditions** emulated from real measurements.

Every script works on a **copy** of `data/app.db`, under `evaluation/results/_work/`. It never writes to the real
database, to the experience memory, or to the project's Langfuse.

---

## 1. Status (as of 1 Nov 2026)

| # | Result | Script | Status |
|---|---|---|---|
| 1 | Latency breakdown per step | `run_answer_matrix` → `analyze_results` | ⏳ matrix 33/75 questions |
| 2 | Real network profiles | `network_profile` | ⏳ only `benchmark-base` measured; other networks still to do |
| 3 | Edge vs cloud under each network | `replay_adaptation` | ⏳ needs the full matrix |
| 4 | Adaptability under a changing network (engine v1 vs v2) | `replay_adaptation` | ⏳ needs the full matrix |
| 5 | CRAG vs RAG: accuracy (MCQ) and faithfulness (doubts) | `run_answer_matrix`, `--set doubts` | ⏳ MCQ part in matrix; doubts not run |
| 6 | Video recommendation strategies R1/R2/R3 | `bench_videos` | ❌ first run invalid (YouTube block); re-run |
| 7 | Speech: WER and latency | `bench_speech` | ✅ done (synthetic voices); real recordings optional |
| 8 | Content validity | `content_stats` | ✅ done |
| 9 | Cost of each coach action | `coach_costs` | ✅ local steps done; scaffold part not run |
| 10 | EWM weights and weight sensitivity | `ewm_sensitivity` | ⏳ needs the replay |

Results so far are in `evaluation/results/`. Anything under `_work/` is scratch: pilots, database copies, and
the invalid video run in `_work/videos_blocked_2026-10-31/`. **Never quote numbers from `_work/`.**

---

## 2. Rules

1. **One laptop per experiment.** Latency, energy and network timings depend on the machine.
   - The **answer matrix** (results 1, 3, 4, part of 5) must finish on the laptop that started it, which is Vani's.
   - The **doubts set** (result 5b) and the **scaffold experiment** (result 9b) are self-contained and can run on
     another laptop. If they do, write that laptop's CPU, GPU, RAM and network in the report next to those
     two tables.
2. **Keep the app closed** while a benchmark runs: no Streamlit, no FastAPI, and nothing heavy on the GPU.
3. **Use your own Groq key in your own `.env`.**
   - Never share keys, and never put two people's keys on one machine to get around the limit (against Groq's
     terms).
   - The free tier allows **200,000 tokens per day** and **8,000 per minute** for `openai/gpt-oss-120b`. The
     scripts wait out both automatically.
4. **YouTube blocks transcript requests** (`IpBlocked`) after too many in a short time.
   - Run `bench_videos` at most once per day.
   - Use a different network (e.g. a phone hotspot) if the main one was blocked recently.
   - Stop it if the log shows "blocked".
5. **Don't run a speed test while the matrix runs.** It saturates the link and spoils the cloud timings.

---

## 3. Setting up a new machine

```powershell
git clone <repo> ; cd <repo>
git checkout <branch with evaluation/benchmarks>
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -r requirements-speech.txt      # only for bench_speech
.venv\Scripts\python -m pip install matplotlib scipy speedtest-cli nvidia-ml-py
```

**Ollama**, for the local (edge) model. Install it from https://ollama.com, then:

```powershell
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

**`.env`** in the repo root needs:

```
GROQ_API_KEY=<your own key>
YOUTUBE_API_KEY=<your own key>     # only for bench_videos
HF_TOKEN=<read token>              # only for bench_speech (SraVaani is gated: accept its terms on Hugging Face)
```

**Database.** The benchmarks read the content packages from `data/app.db`.

1. Create the database:
   ```powershell
   .venv\Scripts\python scripts\migrate_db.py
   ```
2. Load each topic (dry run first; `--apply` writes):
   ```powershell
   .venv\Scripts\python scripts\load_content_package.py life_processes
   .venv\Scripts\python scripts\load_content_package.py life_processes --apply
   ```
   Repeat for `control_and_coordination`, `how_do_organisms_reproduce`, `heredity` and `our_environment`.
3. If the loader complains about missing chapter evidence (`chapter.txt` is gitignored for copyright), ask Vani for
   a copy of her `data/app.db` instead. Don't commit it.

The lesson notes in `knowledge_prep/samples/` are in git, so the topic indexes build themselves on first use.

**Check the setup:**

```powershell
.venv\Scripts\python -m pytest -q     # all tests should pass
```

Then do a 5-question pilot. It writes to `_work/` only:

```powershell
.venv\Scripts\python -m evaluation.benchmarks.run_answer_matrix --per-topic 1 --kinds post --out evaluation/results/_work/pilot.csv
```

---

## 4. How to run each experiment

Run every command from the repo root. All of them can be stopped and re-run: finished rows are kept and skipped.

### Answer matrix (results 1, 3, 4, 5a). Vani's laptop only

```powershell
.venv\Scripts\python -m evaluation.benchmarks.run_answer_matrix --kinds post --variants P1 P2 P3 P4 P5 RAG-local LLM-local --wait-daily
```

- **What it does:** each post-test question (15 per topic, 75 in all) is answered once by each variant through the
  agent's real QA nodes:

  | Variant | What it is |
  |---|---|
  | P1–P5 | the five runtime policies |
  | RAG-local | P2's retrieval without the CRAG retry |
  | LLM-local | the edge model without retrieval (closed book) |

- **What it records for each answer:**
  - correctness against the answer key;
  - retrieval score and CRAG retries;
  - timings per step;
  - LLM calls and tokens;
  - CPU, memory, and GPU power (NVML);
  - bytes sent to the cloud.
- **Output:** `evaluation/results/answer_matrix.csv`, plus `idle_baseline.jsonl` (idle power, for energy above
  idle).
- **Time:** limited by Groq's daily cap, about 45 questions per day. `--wait-daily` makes it sleep and continue
  by itself.
- **Not in the remaining runs:**
  - `RAG-cloud` and `LLM-cloud` were measured on the first 47 questions only.
  - CRAG's retry never fired on MCQs, so RAG-cloud equals P4 there.
  - Say this in the report.

### Learner doubts, CRAG vs RAG (result 5b). Can run on another laptop

```powershell
.venv\Scripts\python -m evaluation.benchmarks.run_answer_matrix --set doubts --wait-daily
```

- **Questions:** 30 learner doubts and 10 off-topic questions.
- **Variants:** RAG-local, P2, RAG-cloud and P4.
- **Records:**
  - retrieval score;
  - whether the CRAG retry fired;
  - whether the answer declined ("not covered");
  - **faithfulness**: the share of claims supported by the retrieved text, judged by `gpt-oss-20b`.
- **Output:** `evaluation/results/doubts_matrix.csv`.
- **Note for the report:** the 10 off-topic questions were written for this benchmark (see `harness.py`, `OFF_TOPIC`).

### Coach costs (result 9)

```powershell
.venv\Scripts\python -m evaluation.benchmarks.coach_costs            # local steps + scaffold experiment (uses Groq)
.venv\Scripts\python -m evaluation.benchmarks.coach_costs --no-llm   # local steps only
```

- **Local steps:** timings for planning, coach decisions, picking formative items and scoring the post-test.
- **Scaffold experiment:** the 30 doubts answered by P1 and P3 in three styles (plain, step-by-step, and
  step-by-step with a hint first), comparing latency, tokens and answer length.
- **Output:** `evaluation/results/coach/`.
- The scaffold part can run on another laptop.

### Network profiles (result 2). Run on each network you want in the report

```powershell
.venv\Scripts\python -m evaluation.benchmarks.network_profile --label college-wifi
.venv\Scripts\python -m evaluation.benchmarks.network_profile --label phone-hotspot-4g
.venv\Scripts\python -m evaluation.benchmarks.network_profile --label weak-signal --no-speedtest
```

- **Measures:**
  - 50 TCP connects to `api.groq.com`, giving RTT median and p95, jitter, and loss;
  - the app's own HTTP probe;
  - a speedtest.net download and upload (skip it with `--no-speedtest`).
- **Output:** one row per run in `evaluation/results/network_profiles.csv`. Run each network 2–3 times at
  different hours.
- **To use them in the replay,** map them to profile names:
  `--profiles good=college-wifi moderate=phone-hotspot-4g poor=weak-signal --base benchmark-base`.
  Without that, the replay uses assumed profiles (good 40 ms, moderate 120 ms, poor 600 ms), and the report must
  say they were assumed.

### Replay: edge vs cloud and adaptability (results 3, 4). After the matrix is complete

```powershell
.venv\Scripts\python -m evaluation.benchmarks.replay_adaptation --reps 20 [--profiles ... --base benchmark-base]
```

- **No API calls.** It takes a few minutes and runs on any laptop, as long as it has the finished
  `answer_matrix.csv` and `idle_baseline.jsonl`.
- **Static part (result 3):** every policy under each network profile, plus a sweep of RTT from 20 to 2000 ms that
  finds the crossover where the cloud stops paying off.
- **Dynamic part (result 4):**
  - **Trace:** good → poor → blackhole → moderate → offline → good, 44 sessions of 5 questions, 20 repetitions.
  - **Strategies compared:**
    - fixed P1–P5;
    - an RTT rule;
    - random;
    - **adaptive engine v1** (as shipped) and **v2** (improved), each starting empty ("cold") or warm-started on a
      training trace over the other half of the questions;
    - the oracle.
- **Output:** `evaluation/results/replay/` (`static_profiles.csv`, `rtt_sweep.csv`, `sessions.csv`).

### EWM weights and sensitivity (result 10). After the replay

```powershell
.venv\Scripts\python -m evaluation.benchmarks.ewm_sensitivity
.venv\Scripts\python -m evaluation.benchmarks.replay_adaptation --skip-static --reps 20 --weights-file evaluation/results/ewm/ewm_weights.json --out replay_ewm
.venv\Scripts\python -m evaluation.benchmarks.ewm_sensitivity --compare replay_ewm
```

This produces:
- the entropy weights per group;
- a utility sweep over λ = 0.5 … 0.9 with the strategy ranking and Kendall's τ;
- the whole trace re-run under the EWM weights.

Output: `evaluation/results/ewm/`.

### Speech (result 7). Done; re-run only to add real voices

```powershell
.venv\Scripts\python -m evaluation.benchmarks.bench_speech
```

- **Round trip:** Kokoro speaks each sentence and SraVaani or Whisper transcribes it back.
- **Real voices (stronger evidence):**
  1. Put WAV files in `evaluation/speech_recordings/`.
  2. Add a `transcripts.csv` with the columns `file,language,text`, where `language` is `en` or `hi`.
  3. Re-run the benchmark.
- **Output:** `evaluation/results/speech/`.

### Video recommendations (result 6). Needs a re-run

```powershell
.venv\Scripts\python -m evaluation.benchmarks.bench_videos --gap-s 90
```

- **Runs:** 20 in total, each with an empty cache.
  - R1, R2 and R3 on each topic;
  - R2 on one concept-focused query per topic, which is what "another video" costs.
- **Also measures** the time to make the top video answerable (transcript → index).
- **YouTube quota:** about 100 units per run, about 2,000 in all.
- **Watch the log.** If the index errors say "blocked", stop it and try another day or network.
- **Output:** `evaluation/results/videos/runs.csv` and `items.csv`. `items.csv` has an empty `relevant` column the
  team can fill in (relevant yes or no) to give a human relevance score.

### Content validity (result 8). Done

```powershell
.venv\Scripts\python -m evaluation.benchmarks.content_stats
```

### Tables, statistics and figures. Run last

```powershell
.venv\Scripts\python -m evaluation.benchmarks.analyze_results
```

- **Tables:** `evaluation/results/tables/*.csv` and `*.tex`, as booktabs LaTeX ready for the report.
- **Figures:** `evaluation/results/figures/*.pdf` and `*.png`.
- **Statistics:** `evaluation/results/stats.txt`, with:
  - McNemar tests for accuracy;
  - Friedman, plus Wilcoxon with Holm correction, for the strategies;
  - 95% bootstrap and Wilson intervals;
  - time to adapt;
  - prediction error.

### Everything after the matrix, in one go (unattended)

```powershell
$py = ".venv\Scripts\python"
& $py -m evaluation.benchmarks.run_answer_matrix --kinds post --variants P1 P2 P3 P4 P5 RAG-local LLM-local --wait-daily
& $py -m evaluation.benchmarks.run_answer_matrix --set doubts --wait-daily     # skip if run on another laptop
& $py -m evaluation.benchmarks.coach_costs                                     # skip if run on another laptop
& $py -m evaluation.benchmarks.replay_adaptation --reps 20
& $py -m evaluation.benchmarks.ewm_sensitivity
& $py -m evaluation.benchmarks.replay_adaptation --skip-static --reps 20 --weights-file evaluation/results/ewm/ewm_weights.json --out replay_ewm
& $py -m evaluation.benchmarks.ewm_sensitivity --compare replay_ewm
& $py -m evaluation.benchmarks.analyze_results
```

If results come from two laptops, copy the other laptop's `doubts_matrix.csv` and `coach/scaffold.csv` into
`evaluation/results/` on the main one before `analyze_results`.

---

## 5. Files in this folder

| File | What it is |
|---|---|
| `isolation.py` | Copies the database to `_work/<run>/` and points the app at it; disables tracing. Every script calls it first. |
| `harness.py` | Question loading, the variants, `answer()` (runs the real QA nodes and measures them), resource sampler, MCQ answer parsing. |
| `run_answer_matrix.py` | Resumable runner for the MCQ matrix and the doubts set; handles Groq rate limits and the daily cap. |
| `network_profile.py` | Real network measurement (result 2). |
| `network_model.py` | How a network profile changes a cloud call. Assumes 4 round trips per call, transfer time, and lost packets costing a retransmission timeout (RTO). Outages are "offline" (fails fast) or "blackhole" (waits for the 60 s timeout). |
| `replay_adaptation.py` | Results 3 and 4: replays the matrix under network profiles; runs the real `PolicyEngine` + `ExperienceStore` (v1 and v2). |
| `ewm_sensitivity.py` | Result 10. |
| `coach_costs.py`, `bench_speech.py`, `bench_videos.py`, `content_stats.py` | Results 9, 7, 6 and 8. |
| `analyze_results.py` | Tables, statistics and figures. |

---

## 6. Must be written in the report

- **Network conditions:** emulated in the application from measured (or, where stated, assumed) profiles. Every
  answer, timing and resource figure is real; the network model only adds what a profile does to the cloud calls.
  The model has not yet been checked against live injected delays; until it is, call it a limitation.
- **Benchmark network:** the matrix was measured with a 28.8 ms median RTT to Groq (`benchmark-base`).
- **Question set:** 75 post-test items (15 per topic, balanced). The extra pre-test items are additional data.
- **Content status:**
  - the items were AI-generated from the NCERT text and are human-review **pending**;
  - 28 of the 60 review flags are still open (result 8).
- **QoE coverage:** no learners, so the EQ and UX groups (learning gain, ratings) are absent. QoE is renormalised
  over AQ and NQ, and `qoe_coverage` is reported.
- **Faithfulness:** judged by an LLM (`gpt-oss-20b`). Spot-check about 20 answers by hand and report the agreement.
- **Speech:** a synthetic-voice round trip is a lower bound for WER. The Hindi test sentences were written for the
  benchmark.
- **Adaptive engine:**
  - v1 is the shipped engine;
  - v2 is the improvement: per-policy neighbours, log RTT, UCB exploration, a local-only guard, and a 4 s probe;
  - the app still defaults to v1 (`POLICY_ENGINE=v1`).

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `rate limited, waiting …` | Normal (8,000 tokens per minute); it continues by itself. |
| `daily token cap reached, waiting 15 min` | Normal with `--wait-daily`; the cap frees up over 24 h. Leave the laptop on. |
| `giving up for now` | Run without `--wait-daily` and the daily cap was hit: re-run the same command later, finished rows are kept. |
| CSV "Expected N fields, saw N+1" | An old run appended a different column set. Keep a backup, then drop the extra last field from those rows (see `_work/answer_matrix.backup.csv`). The runner now keeps the existing header. |
| `KnowledgeUnavailable: No lesson notes` | `knowledge_prep/samples/<topic>.txt` is missing: pull the repo again. |
| `IpBlocked` / "transcript blocked" | YouTube is blocking this network. Stop video work; retry after some hours or on another network. |
| SraVaani 403 / gated | Accept its terms on Hugging Face and use a token allowed to read gated repos. |
| Very slow local answers | Ollama isn't on the GPU (`ollama ps` should say `GPU`), or the app or another GPU job is running. |
