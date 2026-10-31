"""Result 8: content validity of the five topic packages (assessment items, concept tags, review decisions).

    .venv\\Scripts\\python -m evaluation.benchmarks.content_stats

Reads knowledge_prep/packages/<topic>/{manifest,validation,review_log}.json and evaluation/review/flagged_items.csv;
writes evaluation/results/content/{content_summary.csv, flags.csv, content_stats.txt}. Nothing is generated or
judged here: the blind-model checks were run earlier by validate_content_package.py --llm, and every flag was
decided by the project team (a "keep" resolves the flag, it does not approve the item).
"""
import csv
import json
from collections import Counter
from pathlib import Path

from scipy.stats import chisquare

REPO = Path(__file__).resolve().parents[2]
PACKAGES = REPO / "knowledge_prep" / "packages"
OUT = REPO / "evaluation" / "results" / "content"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, lines = [], []
    for pkg in sorted(p for p in PACKAGES.iterdir() if (p / "manifest.json").exists()):
        manifest = json.loads((pkg / "manifest.json").read_text(encoding="utf-8"))
        validation = json.loads((pkg / "validation.json").read_text(encoding="utf-8"))
        review = json.loads((pkg / "review_log.json").read_text(encoding="utf-8")) if (pkg / "review_log.json").exists() else {}
        s, llm = validation["summary"], validation.get("llm_checks") or {}
        positions = Counter()
        for kind in ("pretest", "posttest", "formative"):
            positions.update({int(k): v for k, v in s["answer_positions"][kind].items()})
        counts = [positions.get(i, 0) for i in range(4)]
        checked = llm.get("checked") or 0
        decisions = Counter(d["decision"].split()[0].lower() for d in review.get("decisions", []) if d.get("decision"))
        rows.append({
            "topic_id": manifest["topic_id"], "chapter": manifest.get("chapter"),
            "pages": "-".join(map(str, manifest["source"].get("pages", []))), "review_status": manifest["review_status"],
            "concepts": s["concepts"], "pretest": s["pretest"], "posttest": s["posttest"], "formative": s["formative"],
            "items": s["pretest"] + s["posttest"] + s["formative"],
            "easy/medium/hard (pre+post)": "/".join(str(s["difficulty"]["pretest"].get(k, 0) + s["difficulty"]["posttest"].get(k, 0))
                                                    for k in ("easy", "medium", "hard")),
            "answer_positions_ABCD": "/".join(map(str, counts)),
            "position_chi2_p": round(chisquare(counts).pvalue, 3),
            "max_verbatim_run_words": s["verbatim"]["max_run"],
            "blind_checked": checked,
            "key_agreement": round(1 - len(llm.get("answer_disagreements", [])) / checked, 3) if checked else None,
            "tag_agreement": round(1 - len(llm.get("concept_disagreements", [])) / checked, 3) if checked else None,
            "flags_decided": sum(decisions.values()), **{f"decision_{k}": v for k, v in sorted(decisions.items())},
            "videos_examined": s.get("videos", {}).get("examined"), "video_candidates": s.get("videos", {}).get("candidates"),
            "concepts_with_video": len(s.get("videos", {}).get("concepts_covered", [])),
            "warnings": len(validation.get("warnings", [])),
        })
    with open(OUT / "content_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), restval=0)
        writer.writeheader()
        writer.writerows(rows)

    flags = REPO / "evaluation" / "review" / "flagged_items.csv"
    flag_rows = list(csv.DictReader(open(flags, encoding="utf-8-sig"))) if flags.exists() else []
    decision_col = next((c for c in (flag_rows[0] if flag_rows else {}) if c.startswith("decision")), None)
    by_flag = Counter((r["flag"], (r.get(decision_col) or "open").split()[0].lower() or "open") for r in flag_rows)

    total = {k: sum(r[k] for r in rows) for k in ("concepts", "pretest", "posttest", "formative", "items", "blind_checked")}
    lines.append(f"Topics: {len(rows)}; concepts {total['concepts']}; items {total['items']} "
                 f"(pre {total['pretest']}, post {total['posttest']}, formative {total['formative']})")
    agree = [(r["key_agreement"], r["blind_checked"]) for r in rows if r["key_agreement"] is not None]
    tags = [(r["tag_agreement"], r["blind_checked"]) for r in rows if r["tag_agreement"] is not None]
    if agree:
        lines.append(f"Blind-model answer-key agreement (pooled): "
                     f"{sum(a * n for a, n in agree) / sum(n for _, n in agree):.3f} over {sum(n for _, n in agree)} items")
        lines.append(f"Blind-model concept-tag agreement (pooled): "
                     f"{sum(a * n for a, n in tags) / sum(n for _, n in tags):.3f}")
    lines.append(f"Flags in the review sheet: {len(flag_rows)}; by (flag, decision): {dict(sorted(by_flag.items()))}")
    lines.append("Review status of every package: " + ", ".join(sorted({r['review_status'] for r in rows})))
    lines.append("Answer-position balance: chi-square p per topic " + ", ".join(
        f"{r['topic_id']} {r['position_chi2_p']}" for r in rows) + " (p > 0.05: no detectable position bias)")
    (OUT / "content_stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
