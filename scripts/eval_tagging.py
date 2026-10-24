"""Tagging accuracy of matching.match_question against hand-labelled questions (evaluation/tagging/questions.csv).

  .venv\\Scripts\\python scripts\\eval_tagging.py --concepts   # write the concept-id sheet for labellers
  .venv\\Scripts\\python scripts\\eval_tagging.py              # score the labelled rows (needs Ollama for embeddings)

Labelling (do it BEFORE looking at any prediction): for each row write the concept id the question is about in
gold_concept_id, or "none" if it is about no concept of the topic (off-topic). A second labeller may fill
gold_concept_id_2 independently; their agreement is reported. Unlabelled rows are skipped and counted.

Reported, at COACH_MATCH_MIN_CONFIDENCE (the threshold above which the coach records a tag):
- accuracy: the prediction is right (for a "none" question: nothing is recorded);
- coverage: share of on-topic questions whose tag is recorded;
- precision: share of recorded tags that are right (wrong recorded tags corrupt the tracker);
- a threshold sweep, for choosing the threshold; per-topic accuracy; every error.
Results also go to evaluation/tagging/results.json.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matching  # noqa: E402
from config import PROJECT_ROOT, settings  # noqa: E402
from knowledge_prep.concepts import load_vocabulary  # noqa: E402

DIR = PROJECT_ROOT / "evaluation" / "tagging"
SHEET, RESULTS, CONCEPTS = DIR / "questions.csv", DIR / "results.json", DIR / "concept_ids.md"
NONE = "none"


def load_rows(path: Path = SHEET) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def validate(rows: list[dict[str, str]]) -> list[str]:
    """Labels that are not a concept id of the row's topic (or "none")."""
    problems = []
    for row in rows:
        vocabulary = load_vocabulary(row["topic_id"])
        ids = {c.id for c in vocabulary.concepts} if vocabulary else set()
        for column in ("gold_concept_id", "gold_concept_id_2"):
            label = row.get(column, "")
            if label and label != NONE and label not in ids:
                problems.append(f"{row['id']}: {column} {label!r} is not a concept of {row['topic_id']}")
    return problems


def score(rows: list[dict[str, str]], predictions: dict[str, tuple[str | None, float]], threshold: float) -> dict:
    """Metrics over labelled rows (see the module docstring)."""
    labelled = [r for r in rows if r["gold_concept_id"]]
    on_topic = [r for r in labelled if r["gold_concept_id"] != NONE]
    correct, recorded, recorded_right, errors = 0, 0, 0, []
    for r in labelled:
        concept, confidence = predictions[r["id"]]
        is_recorded = concept is not None and confidence >= threshold
        gold = r["gold_concept_id"]
        right = (not is_recorded) if gold == NONE else (is_recorded and concept == gold)
        correct += right
        if is_recorded:
            recorded += 1
            recorded_right += concept == gold
        if not right:
            errors.append({"id": r["id"], "question": r["question"], "gold": gold, "predicted": concept,
                           "confidence": confidence, "recorded": is_recorded})
    recorded_on_topic = sum(1 for r in on_topic if predictions[r["id"]][0] is not None
                            and predictions[r["id"]][1] >= threshold)
    return {"threshold": threshold, "labelled": len(labelled), "on_topic": len(on_topic),
            "accuracy": _rate(correct, len(labelled)), "coverage": _rate(recorded_on_topic, len(on_topic)),
            "precision": _rate(recorded_right, recorded), "recorded": recorded, "errors": errors}


def agreement(rows: list[dict[str, str]]) -> dict | None:
    """Raw agreement and Cohen's kappa between the two labellers, on rows both labelled."""
    both = [r for r in rows if r["gold_concept_id"] and r.get("gold_concept_id_2")]
    if not both:
        return None
    observed = sum(r["gold_concept_id"] == r["gold_concept_id_2"] for r in both) / len(both)
    labels = {r["gold_concept_id"] for r in both} | {r["gold_concept_id_2"] for r in both}
    expected = sum((sum(r["gold_concept_id"] == lab for r in both) / len(both))
                   * (sum(r["gold_concept_id_2"] == lab for r in both) / len(both)) for lab in labels)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return {"rows": len(both), "agreement": round(observed, 4), "cohens_kappa": round(kappa, 4)}


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def write_concept_sheet() -> Path:
    lines = ["# Concept ids for labelling (evaluation/tagging/questions.csv)", "",
             'Write one id per question in `gold_concept_id`, or `none` if the question is off-topic.', ""]
    for topic_id in sorted({r["topic_id"] for r in load_rows()}):
        vocabulary = load_vocabulary(topic_id)
        lines.append(f"## {topic_id}" + (f" ({vocabulary.status})" if vocabulary else " (NO VOCABULARY)"))
        for c in (vocabulary.concepts if vocabulary else []):
            lines.append(f"- `{c.id}`: {c.name}. {c.description}")
        lines.append("")
    CONCEPTS.write_text("\n".join(lines), encoding="utf-8")
    return CONCEPTS


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Tagging accuracy of matching.match_question.")
    parser.add_argument("--concepts", action="store_true", help="write the concept-id sheet for labellers")
    parser.add_argument("--sheet", type=Path, default=SHEET)
    args = parser.parse_args()
    if args.concepts:
        print(f"Wrote {write_concept_sheet()}")
        return 0
    rows = load_rows(args.sheet)
    if problems := validate(rows):
        print("Fix these labels first:\n  " + "\n  ".join(problems))
        return 1
    labelled = [r for r in rows if r["gold_concept_id"]]
    print(f"{len(rows)} questions, {len(labelled)} labelled, {len(rows) - len(labelled)} not labelled yet.")
    if not labelled:
        print("Label gold_concept_id first (see the module docstring); nothing to score.")
        return 1
    predictions = {r["id"]: matching.match_question(r["question"], r["topic_id"]) for r in labelled}
    main_score = score(rows, predictions, settings.coach_match_min_confidence)
    sweep = [{k: v for k, v in score(rows, predictions, t / 100).items() if k != "errors"}
             for t in range(40, 95, 5)]
    by_topic = {t: score([r for r in rows if r["topic_id"] == t], predictions,
                         settings.coach_match_min_confidence)["accuracy"]
                for t in sorted({r["topic_id"] for r in labelled})}
    result = {**main_score, "by_topic": by_topic, "sweep": sweep, "labeller_agreement": agreement(rows),
              "predictions": {k: {"concept_id": c, "confidence": p} for k, (c, p) in predictions.items()}}
    RESULTS.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nAt threshold {main_score['threshold']}: accuracy {main_score['accuracy']}, coverage "
          f"{main_score['coverage']}, precision of recorded tags {main_score['precision']} "
          f"({main_score['recorded']} recorded)")
    print("Per topic accuracy:", by_topic)
    if result["labeller_agreement"]:
        print("Labeller agreement:", result["labeller_agreement"])
    print("\nthreshold  accuracy  coverage  precision")
    for s in sweep:
        print(f"  {s['threshold']:.2f}     {s['accuracy']}     {s['coverage']}     {s['precision']}")
    print("\nErrors:")
    for e in main_score["errors"]:
        print(f"  {e['id']}: gold {e['gold']}, predicted {e['predicted']} ({e['confidence']:.2f}"
              f"{', recorded' if e['recorded'] else ''}): {e['question']}")
    print(f"\nSaved {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
