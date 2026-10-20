# Concept tracker (LearnerTopicStrength)

This is the first step of the proactive-coach proposal. It records, per learner, how well they know each concept of a
topic. Later coaching nodes (plan session, formative checks, the coach) read it. In the ER diagram it is
**LearnerTopicStrength**.

## Pieces

| Piece | File | What it does |
|---|---|---|
| Concept vocabulary | `knowledge_prep/concepts/<topic_id>.json`, `knowledge_prep/concepts.py` | 6–12 concepts per topic. Each has an id, name, description, aliases, and an **evidence** sentence that must appear verbatim in the lesson notes. Status `draft` → `approved` after human review. |
| Drafting and review | `scripts/extract_concepts.py` | Groq (gpt-oss-120b) drafts from the notes plus the question bank's labels. Drafts are rejected and retried automatically if a concept's evidence isn't in the notes, or if one name or alias points at two concepts. |
| Tracker tables | `app/db/models/concept_tracker.py` | `learner_topic_strength`: one row per learner × topic × concept with attempts, correct, mastery and coverage. `concept_events`: the raw observations, unique per origin. |
| Migration | `app/db/migrations.py`, `scripts/migrate_db.py` | `0001_concept_tracker` creates both tables. `schema_migrations` records what was applied. It also runs on app start. |
| Repository | `app/db/repositories/concept_tracker.py` | `record_attempt`, `mark_covered`, `get_mastery` |
| Recording tests | `agent/concept_tracking.py`, `agent/nodes/assessment.py` | Every scored pre/post-test is recorded live. The backfill uses the same code. |
| Backfill | `scripts/backfill_concept_tracker.py` | Records every stored pre/post-test. Idempotent. `--dry-run` shows the label mapping without writing anything. |

## Mastery

```
mastery = (correct + 1) / (attempts + 2)
```

This is the Laplace (rule-of-succession) estimate:
- **0.5 with no evidence:** "unknown", not "weak".
- **Moves gradually:** one right answer gives 0.67, one wrong answer gives 0.33.
- **Approaches the success rate** as evidence grows: 9 of 10 right gives 0.83.

Coverage (`mark_covered`: the concept came up in a question or a video) is counted separately and never changes
mastery. Exposure isn't evidence of learning.

## Idempotency

Every observation is stored once in `concept_events`. It is keyed by kind (attempt or covered), source (e.g.
`pre-test`), source reference (`<assessment id>:<item index>`) and concept. Recording the same observation again changes
nothing. That covers re-running the backfill, a graph node that runs twice, and live recording followed by the backfill.

## Mapping test items to concepts

Stored test items have free-text concept labels written by the question generator. The matcher tries, in order:
1. the exact name or id;
2. an alias;
3. **nomic-embed-text similarity** of "label + question text" to each concept's name and description. It accepts only
   if similarity is **≥ 0.6** *and* beats the runner-up by **≥ 0.03**. Otherwise the item is left unmapped and reported.

In the dry run on `data/app.db`, "Voluntary Movement" scored `reflex_arc` 0.636 against `plant_tropisms` 0.635, a coin
toss, and the match was wrong (reflexes are involuntary). The margin rule stops this. The thresholds rest on a sample of
**6 embedding cases**; treat them as defaults, not a calibration.

## Commands

```powershell
.venv\Scripts\python scripts\extract_concepts.py --all                     # draft missing vocabularies
.venv\Scripts\python scripts\extract_concepts.py --show life_processes      # review (lists any problems)
.venv\Scripts\python scripts\extract_concepts.py --approve life_processes --by "Name"
.venv\Scripts\python scripts\migrate_db.py --status
.venv\Scripts\python scripts\backfill_concept_tracker.py --dry-run          # mapping report, no writes
.venv\Scripts\python scripts\backfill_concept_tracker.py                    # migrate + backfill
```

Drafting all five topics takes a few minutes. Groq's free tier allows **8,000 tokens per minute** for gpt-oss-120b and a
draft uses about 3,000, so the script waits whenever Groq asks it to.

## Review checklist (before `--approve`)

For each topic, open the JSON (or run `--show`) and check:
1. **Coverage:** every main section of the notes has a concept. (The first life_processes draft had no blood or blood
   vessels; the redraft fixed that.)
2. **No overlaps:** two concepts mustn't mean the same thing. The script catches identical names and aliases, but not
   overlaps in meaning, such as "Animal Hormones" vs "Hormonal Feedback".
3. **Testable:** each concept supports a few multiple-choice questions. Delete vague ones such as `life_processes_overview`.
4. **Aliases:** real synonyms only. Remove ones that are too broad (e.g. "cell division" for binary fission) or wrong.
5. **IDs:** keep them stable once approved. Tracker rows reference them, so renaming an id after backfilling leaves old
   rows on the old id.

## Status (20 Oct 2026)

| Topic | Concepts | Status | Notes for the reviewer |
|---|---|---|---|
| life_processes | 10 | draft | `life_processes_overview` is vague; circulation is merged into one concept |
| control_and_coordination | 11 | draft | Plant hormones other than auxin are missing; check "Animal Hormones" vs "Hormonal Feedback" |
| how_do_organisms_reproduce | 12 | draft | The alias "cell division" is too broad; "Spore protection" is too narrow a name |
| heredity | 10 | draft | Odd aliases ("bisexual reproduction", "meiotic reproduction") |
| our_environment | 11 | draft | Looks clean |

**Backfill:** tested on a copy of `data/app.db`. It recorded 17 attempts from 25 items. The 8 unmapped items were two
ambiguous labels. A re-run recorded 0. The real `data/app.db` hasn't been backfilled; do it **after** approval. Topics
without a vocabulary (the old Class 8 topics: 11 of 16 stored assessments) are skipped.

## Limitations

- **Labels from older banks are vague:** they were written by gemma before the vocabulary existed (e.g. "Heredity ensures
  offspring inherit traits…"), so some matches are weak. The real fix is to tag new questions with vocabulary ids when
  they're generated (pass the vocabulary into the question prompt). That's the natural next step.
- **Tiny data:** the real database has only 5 Class 10 assessments, so there isn't yet enough to calibrate the matcher.
- **Simple mastery:** the Laplace estimate ignores order and forgetting. Bayesian Knowledge Tracing can replace
  `laplace_mastery` later without changing the tables.
