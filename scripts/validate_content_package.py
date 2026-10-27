"""Validate a topic content package (knowledge_prep/packages/<topic_id>/) and write validation.json.

  .venv\\Scripts\\python scripts\\validate_content_package.py life_processes          # offline checks
  .venv\\Scripts\\python scripts\\validate_content_package.py life_processes --llm    # + blind LLM cross-checks

Textbook excerpts are local only: evidence sentences live in evidence.local.json and the chapter in chapter.txt
(both gitignored). Committed files carry section/page references; a verbatim excerpt found in a committed file is an
error, and long verbatim runs in questions/explanations are warned about (see MAX_VERBATIM_RUN).

Errors (the package must not be loaded): schema, counts, unique ids, four distinct options, answer index, unknown
concept or difficulty, page outside the chapter, evidence not found on the cited page of the local chapter text,
the first PRETEST_ITEMS (default MCQ_N_ITEMS) pre-test questions not covering distinct concepts,
post-test items without a valid one-to-one ``parallel_to`` pre-test item of the same concept and difficulty, the
same question in two banks, formative counts per concept, and the concept list failing the app's own vocabulary
checks (knowledge_prep/concepts.py).

Warnings (for the human reviewer, never proof of correctness): near-duplicate questions across banks, answer-position
skew, long evidence excerpts, questions whose wording names another concept's terms but not their own (lexical tag
check), and with --llm: questions a cloud model answered differently from the key when solving them blind, and
questions it tagged with a different concept. Disagreements are flagged for review; agreement does not prove an item
is right.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field, ValidationError, field_validator  # noqa: E402

from config import PROJECT_ROOT  # noqa: E402
from knowledge_prep.concepts import Vocabulary, concept_terms, contains_term, normalise, overlap_problems  # noqa: E402

PACKAGES = PROJECT_ROOT / "knowledge_prep" / "packages"
DIFFICULTIES = ("easy", "medium", "hard")
COUNTS = {"concepts": (8, 10), "pretest": (15, 15), "posttest": (15, 15), "formative_per_concept": (3, 4)}
MAX_EVIDENCE_WORDS = 40
MAX_VERBATIM_RUN = 10  # words copied in a row from the chapter into committed text before a warning
NEAR_DUPLICATE = 0.6  # token Jaccard between two questions' text


class Review(BaseModel):
    status: str
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    notes: str = ""


class Source(BaseModel):
    section: str = Field(min_length=1)
    page: int


class Question(BaseModel):
    id: str = Field(pattern=r"^[A-Z]{2,3}-(PRE|POST|FRM)-\d{2,3}$")
    concept_id: str
    difficulty: str
    question: str = Field(min_length=10)
    options: list[str] = Field(min_length=4, max_length=4)
    answer_index: int = Field(ge=0, le=3)
    explanation: str = Field(min_length=20)
    source: Source
    parallel_to: str | None = None
    review: Review

    @field_validator("options")
    @classmethod
    def _distinct(cls, options: list[str]) -> list[str]:
        if any(not o.strip() for o in options) or len({normalise(o) for o in options}) != 4:
            raise ValueError("options must be four distinct, non-empty strings")
        return options


def compact(text: str) -> str:
    """Normalised text without spaces: robust to PDF extraction splitting words ("autotr ophs")."""
    return normalise(text).replace(" ", "")


def page_blocks(chapter: str) -> dict[int, str]:
    """Printed page -> compacted text, without the running header ("Science96", "Life Processes 97") that would
    otherwise break a sentence continuing from the previous page."""
    parts = re.split(r"^\[p\.(\d+)\]\s*$", chapter, flags=re.MULTILINE)
    blocks = {}
    for i in range(1, len(parts) - 1, 2):
        page, lines = int(parts[i]), parts[i + 1].strip().splitlines()
        if lines and re.fullmatch(rf"\D*\s*{page}", lines[0].strip()):
            lines = lines[1:]
        blocks[page] = compact("\n".join(lines))
    return blocks


def longest_verbatim_run(text: str, chapter_words: list[str], index: dict[str, list[int]]) -> int:
    """Longest run of consecutive words of `text` that also occurs consecutively in the chapter."""
    words, best = normalise(text).split(), 0
    for start in range(len(words)):
        for pos in index.get(words[start], ()):
            n = 0
            while start + n < len(words) and pos + n < len(chapter_words) and words[start + n] == chapter_words[pos + n]:
                n += 1
            best = max(best, n)
    return best


def tokens(text: str) -> set[str]:
    return {w for w in normalise(text).split() if len(w) > 2}


def jaccard(a: str, b: str) -> float:
    x, y = tokens(a), tokens(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def load(topic_dir: Path, name: str) -> Any:
    return json.loads((topic_dir / name).read_text(encoding="utf-8"))


def validate(topic_id: str) -> dict[str, Any]:
    topic_dir = PACKAGES / topic_id
    errors: list[str] = []
    warnings: list[str] = []
    manifest = load(topic_dir, "manifest.json")
    chapter_path, evidence_path = topic_dir / "chapter.txt", topic_dir / "evidence.local.json"
    chapter = chapter_path.read_text(encoding="utf-8") if chapter_path.exists() else ""
    pages = page_blocks(chapter) if chapter else {}
    evidence = load(topic_dir, "evidence.local.json") if evidence_path.exists() else None
    if not pages or evidence is None:
        errors.append("local chapter.txt / evidence.local.json missing: grounding cannot be checked "
                      "(run scripts/extract_chapter_text.py and the package builder on this machine)")
    evidence = evidence or {"concepts": {}, "items": {}}
    chapter_words = normalise(re.sub(r"^\[p\.\d+\]\s*$", " ", chapter, flags=re.MULTILINE)).split()
    word_index: dict[str, list[int]] = {}
    for i, w in enumerate(chapter_words):
        word_index.setdefault(w, []).append(i)
    first, last = manifest["source"]["pages"]

    # concepts: the app's own vocabulary model and checks, plus package-only fields
    raw_concepts = load(topic_dir, "concepts.json")
    try:
        vocabulary = Vocabulary.model_validate({k: v for k, v in raw_concepts.items()} | {
            "concepts": [{k: v for k, v in c.items() if k != "source"} for c in raw_concepts["concepts"]]})
    except ValidationError as exc:
        errors.append(f"concepts.json does not fit the app's vocabulary schema: {exc.errors()[0]['msg']}")
        vocabulary = None
    concept_ids = [c["id"] for c in raw_concepts["concepts"]]
    lo, hi = COUNTS["concepts"]
    if not lo <= len(concept_ids) <= hi:
        errors.append(f"{len(concept_ids)} concepts; expected {lo}-{hi}")
    if vocabulary is not None:
        errors += [f"concepts: {p}" for p in overlap_problems(vocabulary)]
        if vocabulary.status != "draft":
            errors.append("concepts.json must stay 'draft' until a human review")
    for c in raw_concepts["concepts"]:
        p1, p2 = c["source"]["pages"]
        if not first <= p1 <= p2 <= last:
            errors.append(f"concept {c['id']}: pages {p1}-{p2} outside the chapter ({first}-{last})")
        quote = evidence["concepts"].get(c["id"])
        if pages and not quote:
            errors.append(f"concept {c['id']}: no local evidence excerpt")
        elif pages and not any(compact(quote) in pages.get(p, "") for p in range(p1, p2 + 1)):
            errors.append(f"concept {c['id']}: evidence not found on pages {p1}-{p2}")
        if pages and len(c["evidence"]) > 30 and any(compact(c["evidence"]) in b for b in pages.values()):
            errors.append(f"concept {c['id']}: committed 'evidence' is a verbatim excerpt; keep it in evidence.local.json")

    # questions
    banks: dict[str, list[Question]] = {}
    for kind in ("pretest", "posttest", "formative"):
        data = load(topic_dir, f"{kind}.json")
        items = []
        for raw in data["items"]:
            if "evidence" in raw:
                errors.append(f"{raw.get('id', '?')}: committed file contains an 'evidence' excerpt; move it to "
                              "evidence.local.json")
            try:
                items.append(Question.model_validate(raw))
            except ValidationError as exc:
                err = exc.errors()[0]
                errors.append(f"{raw.get('id', '?')}: {'.'.join(map(str, err['loc']))}: {err['msg']}")
        banks[kind] = items
    all_items = [q for items in banks.values() for q in items]
    ids = Counter(q.id for q in all_items)
    errors += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    prefix = {"pretest": "PRE", "posttest": "POST", "formative": "FRM"}
    for kind, items in banks.items():
        errors += [f"{q.id}: id does not match its bank ({kind})" for q in items if f"-{prefix[kind]}-" not in q.id]
    for q in all_items:
        if q.concept_id not in concept_ids:
            errors.append(f"{q.id}: unknown concept_id {q.concept_id!r}")
        if q.difficulty not in DIFFICULTIES:
            errors.append(f"{q.id}: difficulty {q.difficulty!r} not in {DIFFICULTIES}")
        if q.review.status != "pending" or q.review.reviewed_by:
            errors.append(f"{q.id}: must be pending review with no reviewer")
        quote = evidence["items"].get(q.id)
        if not first <= q.source.page <= last:
            errors.append(f"{q.id}: page {q.source.page} outside the chapter ({first}-{last})")
        elif pages and not quote:
            errors.append(f"{q.id}: no local evidence excerpt")
        elif pages and compact(quote) not in pages[q.source.page] + pages.get(q.source.page + 1, ""):
            errors.append(f"{q.id}: evidence not found on p.{q.source.page} (or the next page)")
        if quote and len(quote.split()) > MAX_EVIDENCE_WORDS:
            warnings.append(f"{q.id}: evidence excerpt has {len(quote.split())} words (keep excerpts short)")
        if kind_of(q) != "posttest" and q.parallel_to is not None:
            errors.append(f"{q.id}: only post-test items may have parallel_to")

    for kind in ("pretest", "posttest"):
        lo, hi = COUNTS[kind]
        if not lo <= len(banks[kind]) <= hi:
            errors.append(f"{kind}: {len(banks[kind])} items; expected {lo}-{hi}")
        missing = sorted(set(concept_ids) - {q.concept_id for q in banks[kind]})
        if missing:
            warnings.append(f"{kind}: no item on {missing}")
    from app.core.config import get_settings

    shown = get_settings().pretest_size
    head = [q.concept_id for q in banks["pretest"][:shown]]
    if len(set(head)) != len(head):
        errors.append(f"pretest: the first {shown} questions (what a new learner is shown) repeat a concept: {head}")
    per_concept = Counter(q.concept_id for q in banks["formative"])
    lo, hi = COUNTS["formative_per_concept"]
    errors += [f"formative: {per_concept.get(c, 0)} items on {c}; expected {lo}-{hi}"
               for c in concept_ids if not lo <= per_concept.get(c, 0) <= hi]

    # parallel forms: one-to-one, same concept and difficulty
    pre = {q.id: q for q in banks["pretest"]}
    targets = Counter(q.parallel_to for q in banks["posttest"])
    for q in banks["posttest"]:
        target = pre.get(q.parallel_to or "")
        if q.parallel_to is None or target is None:
            errors.append(f"{q.id}: parallel_to {q.parallel_to!r} is not a pre-test item")
            continue
        if q.parallel_to == q.id:
            errors.append(f"{q.id}: parallel_to references itself")
        if targets[q.parallel_to] > 1:
            errors.append(f"{q.id}: {q.parallel_to} is the parallel of more than one post-test item")
        if (q.concept_id, q.difficulty) != (target.concept_id, target.difficulty):
            errors.append(f"{q.id}: concept/difficulty {q.concept_id}/{q.difficulty} differ from {target.id} "
                          f"({target.concept_id}/{target.difficulty})")
        if normalise(q.question) == normalise(target.question):
            errors.append(f"{q.id}: same question as {target.id}")
    errors += [f"{p}: no parallel post-test item" for p in pre if p not in targets]

    # duplicates across and within banks
    seen: dict[str, str] = {}
    for q in all_items:
        key = normalise(q.question)
        if key in seen:
            errors.append(f"{q.id}: same question text as {seen[key]}")
        seen.setdefault(key, q.id)
    near = []
    for i, a in enumerate(all_items):
        for b in all_items[i + 1:]:
            if kind_of(a) != kind_of(b) and a.parallel_to != b.id and b.parallel_to != a.id:
                score = jaccard(a.question, b.question)
                if score >= NEAR_DUPLICATE:
                    near.append({"a": a.id, "b": b.id, "jaccard": round(score, 2)})
    warnings += [f"near-duplicate across banks: {n['a']} ~ {n['b']} ({n['jaccard']})" for n in near]

    # answer positions and difficulty mix
    for kind, items in banks.items():
        positions = Counter(q.answer_index for q in items)
        if items and max(positions.values()) / len(items) > 0.4:
            warnings.append(f"{kind}: answer positions skewed {dict(sorted(positions.items()))}")

    # lexical concept-tag check (no embeddings): does the wording name another concept but not its own?
    terms = {c.id: concept_terms(c) for c in vocabulary.concepts} if vocabulary else {}
    tag_flags = []
    for q in all_items:
        text = normalise(f"{q.question} {q.options[q.answer_index]}")
        named = [cid for cid, ts in terms.items() if any(contains_term(text, t) for t in ts)]
        if named and q.concept_id not in named:
            tag_flags.append({"id": q.id, "assigned": q.concept_id, "named_in_text": named})
    warnings += [f"{f['id']}: tagged {f['assigned']}, wording names {f['named_in_text']}" for f in tag_flags]

    # videos (optional file): API-sourced candidates with a verified transcript and known concept ids
    video_summary = None
    if (topic_dir / "videos.json").exists():
        videos = load(topic_dir, "videos.json")
        candidates = videos.get("candidates", [])
        if not 3 <= len(candidates) <= 5:
            errors.append(f"videos: {len(candidates)} candidates; expected 3-5")
        for c in candidates:
            if c.get("transcript", {}).get("status") != "ok":
                errors.append(f"video {c.get('video_id')}: transcript not verified")
            unknown = [x["concept_id"] for x in c.get("covers", []) if x["concept_id"] not in concept_ids]
            if unknown:
                errors.append(f"video {c.get('video_id')}: unknown concept ids {unknown}")
            if (c.get("review") or {}).get("status") != "pending":
                errors.append(f"video {c.get('video_id')}: must be pending review")
        if videos.get("concepts_not_covered"):
            warnings.append(f"videos: no candidate covers {videos['concepts_not_covered']}")
        embedding = str(videos.get("method", {}).get("embedding_coverage", ""))
        if "not run" in embedding:
            warnings.append("videos: concept coverage is lexical only; the embedding check (Ollama) has not run")
        elif "INCOMPLETE" in embedding:
            summary = videos.get("embedding_coverage_summary", {})
            warnings.append(f"videos: embedding coverage INCOMPLETE ({summary.get('videos_checked', '?')}/"
                            f"{summary.get('videos_total', '?')} videos checked); coverage is otherwise lexical only")
        video_summary = {"candidates": len(candidates), "concepts_covered": videos.get("concepts_covered"),
                         "examined": len(videos.get("examined", []))}
    else:
        warnings.append("videos.json missing")

    # verbatim textbook runs in committed text (questions, options, explanations, concept names/descriptions)
    verbatim = []
    if chapter_words:
        for q in all_items:
            run = longest_verbatim_run(" ".join([q.question, *q.options, q.explanation]), chapter_words, word_index)
            verbatim.append({"id": q.id, "longest_run": run})
        for c in raw_concepts["concepts"]:
            verbatim.append({"id": c["id"], "longest_run": longest_verbatim_run(f"{c['name']} {c['description']}",
                                                                                chapter_words, word_index)})
        warnings += [f"{v['id']}: copies {v['longest_run']} consecutive words from the chapter (paraphrase?)"
                     for v in verbatim if v["longest_run"] >= MAX_VERBATIM_RUN]

    summary = {
        "verbatim": {"max_run": max((v["longest_run"] for v in verbatim), default=None),
                     f"items_with_run_>=_{MAX_VERBATIM_RUN}": sum(v["longest_run"] >= MAX_VERBATIM_RUN
                                                                   for v in verbatim)},
        "videos": video_summary,
        "concepts": len(concept_ids),
        **{kind: len(items) for kind, items in banks.items()},
        "formative_per_concept": dict(per_concept),
        "difficulty": {kind: dict(Counter(q.difficulty for q in items)) for kind, items in banks.items()},
        "answer_positions": {kind: dict(sorted(Counter(q.answer_index for q in items).items()))
                             for kind, items in banks.items()},
        "pre_items_per_concept": dict(Counter(q.concept_id for q in banks["pretest"])),
    }
    return {"topic_id": topic_id, "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "valid": not errors, "errors": errors, "warnings": warnings, "summary": summary,
            "lexical_tag_flags": tag_flags, "near_duplicates": near,
            "evidence_checked": bool(pages) and bool(evidence["items"]),
            "first_shown_pretest_concepts": head,
            "_items": all_items, "_concepts": raw_concepts["concepts"]}


def kind_of(q: Question) -> str:
    return {"PRE": "pretest", "POST": "posttest", "FRM": "formative"}[q.id.split("-")[1]]


# --- blind cross-checks with a cloud model (optional) -------------------------------------------------------------

def llm_checks(items: list[Question], concepts: list[dict[str, Any]], batch: int = 8) -> dict[str, Any]:
    """Ask the cloud model to answer each question and pick its concept without seeing the key or the tag."""
    from agent.deps import make_cloud_client
    from services.llm_clients import GenerationRequest, LLMError

    client = make_cloud_client()
    if not client.is_available():
        return {"ran": False, "reason": "no cloud model configured (GROQ_API_KEY)"}
    catalogue = "\n".join(f"- {c['id']}: {c['name']}: {c['description']}" for c in concepts)
    answers, tags, failures = {}, {}, []
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        listing = "\n\n".join(f"[{q.id}] {q.question}\n" + "\n".join(f"  {'ABCD'[i]}. {o}" for i, o in
                                                                  enumerate(q.options)) for q in chunk)
        prompt = (f"Concepts of NCERT Class 10 Science, chapter '{concepts and 'this chapter'}':\n{catalogue}\n\n"
                  f"Questions:\n{listing}\n\nFor each question, give the single best answer letter as a Class 10 "
                  "biology teacher would, and the one concept id from the list that the question mainly tests. "
                  'Reply with JSON only: {"<question id>": {"answer": "A|B|C|D", "concept_id": "<id>"}, ...}')
        request = GenerationRequest(system="You are a careful NCERT Class 10 biology teacher. Reply with JSON only.",
                                    prompt=prompt, max_tokens=3000, temperature=0.0,
                                    operation="validate-content")
        for attempt in range(6):
            try:
                text = client.generate(request).text
                data = json.loads(text[text.index("{"):text.rindex("}") + 1])
                break
            except LLMError as exc:
                wait = re.search(r"try again in ([\d.]+)s", str(exc))
                if wait is None or attempt == 5:
                    failures.append(f"{chunk[0].id}..{chunk[-1].id}: {str(exc)[:160]}")
                    data = {}
                    break
                time.sleep(float(wait.group(1)) + 1)
            except ValueError as exc:
                failures.append(f"{chunk[0].id}..{chunk[-1].id}: unreadable reply ({exc})")
                data = {}
                break
        for q in chunk:
            got = data.get(q.id) or {}
            answers[q.id] = str(got.get("answer", "")).strip().upper()[:1] or None
            tags[q.id] = got.get("concept_id")
    answer_flags = [{"id": q.id, "key": "ABCD"[q.answer_index], "model_answer": answers[q.id]}
                    for q in items if answers.get(q.id) and answers[q.id] != "ABCD"[q.answer_index]]
    tag_flags = [{"id": q.id, "assigned": q.concept_id, "model_concept": tags[q.id]}
                 for q in items if tags.get(q.id) and tags[q.id] != q.concept_id]
    return {"ran": True, "model": f"{client.label}:{client.model}", "checked": len(items),
            "answered": sum(1 for v in answers.values() if v), "answer_disagreements": answer_flags,
            "concept_disagreements": tag_flags, "failures": failures,
            "note": "A disagreement flags the item for human review; agreement does not prove it correct."}


def items_fingerprint(items: list[Question]) -> str:
    """Hash of what the blind checks look at, to tell whether a stored run still matches the questions."""
    content = [(q.id, q.question, q.options, q.answer_index, q.concept_id) for q in items]
    return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode("utf-8")).hexdigest()


def llm_warnings(checks: dict[str, Any], suffix: str = "") -> list[str]:
    if not checks.get("ran"):
        return [f"blind checks did not run: {checks.get('reason')}{suffix}"]
    warnings = []
    if checks["answered"] < checks["checked"]:
        warnings.append(
            f"blind checks INCOMPLETE: {checks['answered']}/{checks['checked']} questions answered "
            f"({len(checks['failures'])} failed batches, first: {checks['failures'][0] if checks['failures'] else '-'}"
            f"); missing answers are not agreements{suffix}")
    warnings += [f"{f['id']}: blind model answered {f['model_answer']}, key is {f['key']}{suffix}"
                 for f in checks.get("answer_disagreements", [])]
    warnings += [f"{f['id']}: blind model tagged {f['model_concept']}, assigned {f['assigned']}{suffix}"
                 for f in checks.get("concept_disagreements", [])]
    return warnings


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Validate a topic content package.")
    parser.add_argument("topic_id")
    parser.add_argument("--llm", action="store_true", help="also run blind answer/concept checks with the cloud model")
    args = parser.parse_args()
    path = PACKAGES / args.topic_id / "validation.json"
    result = validate(args.topic_id)
    items, concepts = result.pop("_items"), result.pop("_concepts")
    fingerprint = items_fingerprint(items)
    if args.llm:
        result["llm_checks"] = checks = llm_checks(items, concepts)
        checks |= {"ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "items_sha256": fingerprint}
        result["warnings"] += llm_warnings(checks)
    elif path.exists() and (previous := json.loads(path.read_text(encoding="utf-8")).get("llm_checks")):
        # keep the last blind-check run rather than erasing it; say so, and whether the questions changed since
        result["llm_checks"] = {**previous, "carried_over": True}
        result["warnings"] += llm_warnings(previous, " (from the last --llm run)")
        if previous.get("items_sha256") != fingerprint:
            result["warnings"].append("blind checks STALE: the questions changed since the last --llm run (or the run "
                                      "predates this check); rerun with --llm")
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.topic_id}: {'VALID' if result['valid'] else 'INVALID'} | {len(result['errors'])} errors, "
          f"{len(result['warnings'])} warnings")
    print("summary:", json.dumps(result["summary"]))
    for e in result["errors"]:
        print("  ERROR  ", e)
    for w in result["warnings"]:
        print("  warning", w)
    print(f"wrote {path}")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
