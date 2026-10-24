# Coaching data contract: matching, video coverage, formative pool, reflection

These are the real implementations behind the coaching layer's data contract (`agent/coach_data.py`). They replace the
stubs that let the coaching graph run before they existed. Signatures are unchanged.

| Function | File | What it returns |
|---|---|---|
| `match_question(text, topic_id)` | `matching.py` | `(concept_id, confidence)`; `(None, 0.0)` for an empty question, or one naming nothing when no embedder is available |
| `video_coverage(video_id, topic_id)` | `matching.py` | Concept ids the video's transcript covers, in vocabulary order |
| `get_formative_items(topic_id, concept_id, n, exclude_ids)` | `formative_pool.py` | Up to `n` practice questions on the concept |
| `render(reflection, names)` | `agent/reflection_template.py` | The learner-facing reflection summary |

## match_question

1. **The question names exactly one concept** (its name or an alias, as whole words; e.g. "What do the villi do?"):
   that concept, with confidence `MATCH_LEXICAL_CONFIDENCE` (0.9).
2. **Otherwise, embedding similarity** (nomic-embed-text, the embedder the agent already has loaded) between the question
   and each concept's text. When the question names several concepts, only those are compared.
   **Confidence** = cosine similarity, scaled down linearly when the runner-up is within `CONCEPT_MATCH_MIN_MARGIN`
   (0.03). An ambiguous question therefore falls below the coach's recording threshold `COACH_MATCH_MIN_CONFIDENCE` (0.6)
   and isn't recorded. Evidence: in the backfill, "Voluntary Movement" scored `reflex_arc` 0.636 against
   `plant_tropisms` 0.635.
3. **No embedder** (offline fakes): one named concept gives 0.9, several named concepts give 0.45, none gives
   `(None, 0.0)`.

## video_coverage

- **No re-embedding:** it uses the transcript chunk vectors already stored in the video's FAISS index (`yt_<video_id>`).
- **A concept is covered when:**
  - a chunk is at least `COVERAGE_MIN_SIMILARITY` (0.70, **provisional**, see Calibration) similar to the concept's
    text, *or*
  - the transcript names the concept at least `COVERAGE_MIN_MENTIONS` (2) times.
- **Not indexed yet** (the video wasn't chosen or prefetched): it raises `CoverageUnknownError`. The coaching code
  already treats an exception as "unknown", so the video card shows no "Covers:" line, rather than a false
  "covers none of your concepts".
- **Caching:** results are cached per video, topic and vocabulary version.

## Formative pool (`formative_items` table, migration `0003_formative_pool`)

**Why a separate pool, not `extend_topic_items`:**
- **Any bank item can be a pre- or post-test question,** and `plan_formative` already excludes every bank item.
  Questions appended to the bank with `extend_topic_items` would therefore always be excluded.
- **Contamination:** a practice question that later became a pre/post-test question would rehearse the test and inflate
  learning gain, the main outcome metric.

**What's reused:** the pool uses `extend_topic_items`' generator (`app.services.mcq.generate_concept_items`: focused
retrieval and prompt, top-up temperature, an avoid-list of the bank and the pool), but stores the questions in
`formative_items`.

**What a generated question must pass:**
- **New:** not already in the bank or pool (compared as normalised text).
- **On the right concept:**
  - its own concept label names the target concept → kept (`label`);
  - its label names **another** concept → dropped;
  - otherwise `match_question` on its text maps it to the target concept with ≥ 0.6 → kept (`matched`); to another
    concept → dropped; no confident match → kept as `unverified`.
- **Stable ids** (`fp_…`) and a per-concept `position`, so the pool is served in a fixed order.

**When the pool runs short:** a request that finds fewer than `n` usable questions tops up to
`FORMATIVE_POOL_TARGET` (4) per concept, at most 3 LLM calls. If the top-up fails, whatever exists is served; with
nothing at all, the error is raised and the coach shows "no practice questions".

**Pre-building:** run `scripts/build_formative_pool.py --all` so learners never wait. It uses **Groq only**: the 1B local
model writes wrong answer keys too often. It waits out Groq's free-tier rate limit (8,000 tokens/min).

## Reflection template

`render()` writes 2–4 sentences:
- goals met, named;
- the biggest mastery changes, as before → after;
- any concept that slipped;
- practice done (checks, suggestions taken);
- what to start with next time.

It's deterministic, with no LLM call. The reflection card lists every goal and concept below it, so the paragraph only
highlights. `build_reflection` falls back to the earlier `summary_text` if rendering ever fails.

## Tagging accuracy check (for the report)

1. Run `scripts/eval_tagging.py --concepts` to write `evaluation/tagging/concept_ids.md` (the valid ids per topic).
2. **Label `evaluation/tagging/questions.csv`** (30 learner questions, 6 per topic) **before looking at any
   prediction:** put the concept id in `gold_concept_id`, or `none` if the question is off-topic. A second person
   labels `gold_concept_id_2` independently.
3. Run `scripts/eval_tagging.py` (needs Ollama). It reports and saves `results.json` with:
   - **accuracy** at the 0.6 threshold;
   - **coverage:** share of on-topic questions tagged;
   - **precision:** share of recorded tags that are correct (wrong tags corrupt the tracker);
   - a **threshold sweep** (0.40–0.90) to choose the threshold;
   - per-topic accuracy, every error, and **Cohen's kappa** between the two labellers.

The sheet mixes question types on purpose:
- named concepts ("villi");
- paraphrases with no concept words ("Why do my legs hurt after running?");
- two-concept questions ("food chain vs food web");
- off-topic questions ("Who discovered gravity?").

## Calibration (pending: Ollama needs reinstalling)

`COVERAGE_MIN_SIMILARITY` (0.70) is a provisional default. Once Ollama works, calibrate it on indexed videos with known
content: for each concept, compare the best chunk similarity on a video that teaches it with one that doesn't. Do this
before reporting coverage numbers. The question threshold (0.6) is calibrated by the tagging sweep above.

## Tests

`tests/test_coach_data_contract.py` (15 tests) covers:
- the lexical, embedding, ambiguous and offline paths of `match_question`;
- unknown, mention and vector coverage, plus caching;
- pool generation outside the bank, rejection of off-concept and duplicate questions, `exclude_ids`, failure fallback
  and contract errors;
- the template variants;
- the evaluation metrics and Cohen's kappa.

These tests use a real FAISS store with a deterministic stand-in embedder, a real in-memory database and the real
generator with a fake LLM.

**Changes to existing tests:**
- **Coaching-flow tests now use a `fixed_formative_items` fixture** (`tests/conftest.py`), so flow assertions don't
  depend on generated questions.
- **`coach_data.implemented()` reflects each function's latest call.** It used to stay False for the rest of the process
  once any call had used the stub, which made one test pass or fail depending on test order.
