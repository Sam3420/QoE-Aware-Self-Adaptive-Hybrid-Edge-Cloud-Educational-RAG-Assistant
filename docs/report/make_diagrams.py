"""Draw the report's design diagrams (Figures 5.1-5.4, 6.1) into docs/report/figures/ as PDF and PNG.

    .venv\\Scripts\\python docs\\report\\make_diagrams.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
INK, MUTED, LINE = "#1f1f1e", "#5f5e58", "#8a8983"
FILL = {"ui": "#dbe8fb", "agent": "#e9e4fb", "learn": "#dcf3ea", "rag": "#fdebd9", "adapt": "#fbe1e9",
        "data": "#efeee9", "ext": "#f6f5f1", "metric": "#fdf3d6"}
NL = "\n"


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, body="", kind="data", size=8.5, top=False):
    """A rounded box with a bold title and a smaller grey body (body lines joined with NL).

    top=True places the body right under the title (tall boxes with long text)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                facecolor=FILL[kind], edgecolor=LINE, linewidth=0.9))
    if not body:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=size, color=INK, linespacing=1.3)
        return (x, y, w, h)
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=size, color=INK, fontweight="bold")
    if top:
        ax.text(x + w / 2, y + h - 9, body, ha="center", va="top", fontsize=size - 1.0, color=MUTED, linespacing=1.5)
    else:
        ax.text(x + w / 2, y + (h - 4.5) / 2, body, ha="center", va="center", fontsize=size - 1.3, color=MUTED,
                linespacing=1.35)
    return (x, y, w, h)


def region(ax, x, y, w, h, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=2", facecolor="none",
                                edgecolor=LINE, linewidth=1.0, linestyle=(0, (4, 3))))
    ax.text(x + 1.2, y + h - 1.4, label, ha="left", va="top", fontsize=9, color=MUTED, fontweight="bold")


def arrow(ax, p, q, text="", dashed=False, rad=0.0, size=7, at=0.5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color=INK,
                                 linestyle=(0, (4, 3)) if dashed else "-", connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))
    if text:
        ax.text(p[0] + (q[0] - p[0]) * at, p[1] + (q[1] - p[1]) * at, text, fontsize=size, color=MUTED, ha="center",
                va="center", bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))


def c(b, side):
    x, y, w, h = b
    return {"t": (x + w / 2, y + h), "b": (x + w / 2, y), "l": (x, y + h / 2), "r": (x + w, y + h / 2)}[side]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220, facecolor="white")
    plt.close(fig)


def lines(*parts):
    return NL.join(parts)


def architecture():
    fig, ax = canvas(11, 7.6)
    region(ax, 1, 2, 70, 96, "EDGE DEVICE (learner's laptop)")
    region(ax, 74, 2, 25, 96, "EXTERNAL SERVICES")
    ui = box(ax, 4, 82, 64, 11, "Learner interface",
             lines("Streamlit + FastAPI · typed or spoken questions · read-aloud answers",
                   "speech-to-text: SraVaani-1.0 (Indian languages + English) · text-to-speech: Kokoro-82M"), "ui")
    ag = box(ax, 4, 64, 64, 12, "LangGraph agent",
             "one state graph per learning session · pauses for the learner (interrupts) · checkpointed in SQLite",
             "agent")
    coach = box(ax, 4, 44, 20, 15, "Proactive coach", lines("mastery tracker", "goals · next-step rules",
                                                             "formative pool · reflection"), "learn")
    rag = box(ax, 26, 44, 20, 15, "Hybrid RAG + CRAG", lines("FAISS index of notes", "+ video transcript",
                                                              "retrieve · grade · retry"), "rag")
    rec = box(ax, 48, 44, 20, 15, "Video recommender", lines("strategies R1 / R2 / R3", "rank by goals",
                                                              "caption check"), "learn")
    pkg = box(ax, 4, 24, 20, 14, "Content packages", lines("concepts · pre/post tests", "formative items · videos"),
              "data")
    edge = box(ax, 26, 24, 20, 14, "Edge LLM", lines("gemma3:1b via Ollama", "on the laptop GPU",
                                                      "embeddings: nomic-embed-text"), "rag")
    router = box(ax, 48, 24, 20, 14, "Router", lines("executes the session's", "policy P1–P5",
                                                      "edge / cloud / escalate"), "rag")
    mon = box(ax, 4, 5, 20, 14, "Monitoring", lines("RTT · CPU · memory · energy", "latency · accuracy",
                                                     "QoE · SRE · U · reward"), "metric")
    mem = box(ax, 26, 5, 20, 14, "Experience memory", lines("past sessions (s, a, R, s′)", "SQLite records",
                                                             "+ FAISS state index"), "data")
    qaps = box(ax, 48, 5, 20, 14, "Adaptive policy engine", lines("QAPS: picks the policy", "from similar past",
                                                                   "sessions (kNN)"), "adapt")
    lf = box(ax, 76, 64, 21, 12, "Langfuse", lines("traces of every node", "and model call (optional)"), "ext")
    yt = box(ax, 76, 44, 21, 15, "YouTube", lines("Data API v3 (search,", "metadata) + caption", "transcripts"), "ext")
    cloud = box(ax, 76, 22, 21, 18, "Cloud LLM (Groq)", lines("gpt-oss-120b: answers", "gpt-oss-20b: grounding",
                                                               "check (P5)"), "ext")
    arrow(ax, c(ui, "b"), c(ag, "t"), "question / answer")
    for b in (coach, rag, rec):
        arrow(ax, (c(b, "t")[0], 64), c(b, "t"))
    arrow(ax, c(ag, "r"), c(lf, "l"), "traces")
    arrow(ax, c(rec, "r"), c(yt, "l"))
    arrow(ax, c(pkg, "t"), c(coach, "b"), "concepts, items")
    arrow(ax, (44, 44), (52, 38), "evidence")
    arrow(ax, c(router, "l"), c(edge, "r"), "edge")
    arrow(ax, c(router, "r"), (76, 31), "cloud")
    arrow(ax, c(mon, "r"), c(mem, "l"), "outcome")
    arrow(ax, c(mem, "r"), c(qaps, "l"))
    arrow(ax, c(qaps, "t"), c(router, "b"), "policy")
    save(fig, "fig5_1_architecture")


def workflow():
    fig, ax = canvas(11.5, 7.4)
    ax.text(1, 99, "Before learning", fontsize=9.5, color=MUTED, fontweight="bold", va="top")
    ax.text(1, 67, "Learning loop", fontsize=9.5, color=MUTED,
            fontweight="bold", va="top")
    ax.text(1, 34, "After learning", fontsize=9.5, color=MUTED, fontweight="bold", va="top")
    w, h = 20.5, 15
    xs = [1.5, 25, 48.5, 72]
    row1 = [box(ax, xs[0], 78, w, h, "1. Start session", lines("student ID + topic", "profile and history"), "ui"),
            box(ax, xs[1], 78, w, h, "2. Pre-test", lines("5 items, one per concept", "missed items retested"), "learn"),
            box(ax, xs[2], 78, w, h, "3. Goals", lines("2–4 weak concepts", "(mastery < 0.5)"), "learn"),
            box(ax, xs[3], 78, w, h, "4. Choose policy", lines("QAPS: measure RTT, CPU,", "memory → P1–P5"), "adapt")]
    row2 = [box(ax, xs[3], 45, w, h, "5. Recommend videos", lines("query from goals", "R1/R2/R3 · caption check"), "learn"),
            box(ax, xs[2], 45, w, h, "6. Learner asks", lines("typed or spoken", "(SraVaani STT)"), "ui"),
            box(ax, xs[1], 45, w, h, "7. Retrieve + answer", lines("CRAG on notes + transcript", "edge / cloud / P5"), "rag"),
            box(ax, xs[0], 45, w, h, "8. Coach next step", lines("check · video · review ·", "follow-up · post-test"), "learn")]
    row3 = [box(ax, xs[0], 12, w, h, "9. Post-test", lines("parallel items", "learning gain g"), "learn"),
            box(ax, xs[1], 12, w, h, "10. Reflection", lines("goals reached · mastery", "weak concepts saved"), "learn"),
            box(ax, xs[2], 12, w, h, "11. Score the session", lines("QoE, SRE, utility U", "reward R"), "metric"),
            box(ax, xs[3], 12, w, h, "12. Store experience", lines("(context, policy, outcome)", "SQLite + FAISS"), "data")]
    for a, b in zip(row1, row1[1:]):
        arrow(ax, c(a, "r"), c(b, "l"))
    for a, b in zip(row2, row2[1:]):
        arrow(ax, c(a, "l"), c(b, "r"))
    for a, b in zip(row3, row3[1:]):
        arrow(ax, c(a, "r"), c(b, "l"))
    arrow(ax, c(row1[3], "b"), c(row2[0], "t"), "policy fixed for the session")
    arrow(ax, (xs[0] + w / 2, 60), (xs[2] + w / 2, 60), "", rad=-0.4)
    ax.text((xs[0] + xs[2]) / 2 + w / 2, 69.5, "next question (at most 10 per session)", fontsize=7, color=MUTED,
            ha="center", bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))
    arrow(ax, c(row2[3], "b"), c(row3[0], "t"), "goals mastered or learner ends")
    arrow(ax, c(row3[3], "r"), c(row1[3], "r"), "", dashed=True, rad=0.3)
    ax.text(99.5, 60, lines("next", "session", "uses this", "experience"), fontsize=7, color=MUTED, ha="right",
            va="center")
    ax.add_patch(FancyBboxPatch((1.5, 1), 91, 5.5, boxstyle="round,pad=0.2,rounding_size=1", facecolor=FILL["metric"],
                                edgecolor=LINE, linewidth=0.8))
    ax.text(47, 3.75, "Monitoring runs throughout: latency, retrieval score, route, CPU / memory / energy, bytes, "
                      "traces (Langfuse)", fontsize=8, color=INK, ha="center", va="center")
    save(fig, "fig5_2_session_workflow")


def coach():
    fig, ax = canvas(11, 5.6)
    s1 = box(ax, 1, 38, 26, 58, "Stage 1 · start of session",
             lines("pre-test (5 items)", "", "update mastery", "m = (correct+1) / (attempts+2)", "",
                   "set 2–4 goals, in order:", "1. still weak from last session", "2. missed in this pre-test",
                   "3. other low mastery", "4. never practised (to reach 2)"), "learn", size=9, top=True)
    s2 = box(ax, 33, 38, 36, 58, "Stage 2 · after every answer",
             lines("tag the question with a concept, update mastery", "",
                   "the first matching rule picks the next step:", "",
                   "1. formative check — goal concept, mastery < 0.5",
                   "2. another video — current video misses a goal",
                   "3. review — answer escalated or weakly grounded",
                   "4. post-test — every goal at mastery ≥ 0.75",
                   "5. follow-up — next unexplored or weakest goal", "",
                   "learner accepts or skips (skipped twice → muted)", "at most 10 questions per session"),
             "learn", size=9, top=True)
    s3 = box(ax, 75, 38, 24, 58, "Stage 3 · end of session",
             lines("parallel post-test", "(same concepts, new items)", "", "learning gain",
                   "g = (post − pre) / (1 − pre)", "", "reflection: goals reached,", "mastery before / after", "",
                   "weak concepts saved", "→ Stage 1 next time"), "learn", size=9, top=True)
    arrow(ax, c(s1, "r"), c(s2, "l"))
    arrow(ax, c(s2, "r"), c(s3, "l"))
    arrow(ax, c(s3, "b"), c(s1, "b"), "", dashed=True, rad=-0.22)
    ax.text(51, 16, "weak concepts carried into the next session", fontsize=7.5, color=MUTED, ha="center")
    save(fig, "fig5_3_proactive_coach")


def qaps():
    fig, ax = canvas(10, 8.4)
    W, H, X = 42, 9, 29
    steps = [("1. Measure context", "probe Groq (4 s timeout) · CPU · memory · competency · last QoE", "data"),
             ("2. Encode the state x(s)", "network delay on a log scale up to the failure value (5,000 ms)", "data"),
             ("3. Feasible policies C", "probe failed → {P1, P2} (edge only); otherwise P1–P5", "data"),
             ("4. Per-policy neighbours", "for each policy: its own k nearest past sessions (FAISS)", "data"),
             ("5. Estimate Û(s, a)", "Gaussian-weighted mean of their utility (σ = 0.25)", "adapt"),
             ("6. Score and choose", "score = Û + c / √(1 + Σw) → highest score (ties → P1)", "adapt"),
             ("7. Run, measure, store", "observed utility U → new experience in memory", "data")]
    boxes, y = [], 88
    for t, b, k in steps:
        boxes.append(box(ax, X, y, W, H, t, b, k, size=8.5))
        y -= 13
    for a, b in zip(boxes, boxes[1:]):
        arrow(ax, c(a, "b"), c(b, "t"))
    notes = [(boxes[2], lines("fewer than 3 stored", "sessions → default P1"), "r"),
             (boxes[3], lines("a policy with no evidence", "near this state → try it"), "r"),
             (boxes[5], lines("with probability ε = 0.05", "→ random feasible policy"), "r")]
    for b, text, where in notes:
        x = 2 if where == "l" else 76
        note = box(ax, x, b[1], 22, H, text, "", "ext", size=7.5)
        arrow(ax, c(b, where), c(note, "r" if where == "l" else "l"), dashed=True)
    arrow(ax, c(boxes[-1], "l"), c(boxes[0], "l"), "", dashed=True, rad=-0.35)
    ax.text(17.5, 50, lines("next", "session"), fontsize=7.5, color=MUTED, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"))
    save(fig, "fig5_4_qaps")


def pipeline():
    fig, ax = canvas(11.5, 4.8)
    q = box(ax, 1, 55, 17, 36, "Question sets", lines("75 keyed post-test MCQs", "(15 per topic)", "30 learner doubts",
                                                       "10 off-topic"), "data")
    m = box(ax, 22, 55, 19, 36, "Answer matrix", lines("each question × each", "policy, real QA nodes:",
                                                        "correct? latency, energy,", "bytes, retrieval, route"), "rag")
    n = box(ax, 22, 6, 19, 36, "Network profiles", lines("measured: college Wi-Fi,", "phone hotspot",
                                                          "assumed: poor,", "offline, blackhole"), "ext")
    nm = box(ax, 45, 6, 19, 36, "Network model", lines("adds a profile's effect", "to each cloud call:",
                                                        "RTTs, jitter, transfer,", "loss, outages"), "ext")
    r = box(ax, 45, 55, 19, 36, "Trace replay", lines("44 sessions × 20 repeats", "good → poor → blackhole",
                                                       "→ moderate → offline", "12 strategies, real engine"), "adapt")
    s = box(ax, 68, 55, 15, 36, "Scoring", lines("QoE, SRE, U", "(app's own code)", "regret vs oracle",
                                                  "time to adapt"), "metric")
    st = box(ax, 86, 55, 13, 36, "Statistics", lines("Friedman", "Wilcoxon + Holm", "McNemar", "bootstrap CI",
                                                      "Kendall τ"), "metric")
    box(ax, 68, 6, 31, 36, "Other benchmarks", lines("speech round trip (WER)", "coach step and scaffolding costs",
                                                     "content validity · video strategies",
                                                     "EWM weight sensitivity"), "data")
    arrow(ax, c(q, "r"), c(m, "l"))
    arrow(ax, c(m, "r"), c(r, "l"))
    arrow(ax, c(n, "r"), c(nm, "l"))
    arrow(ax, c(nm, "t"), c(r, "b"))
    arrow(ax, c(r, "r"), c(s, "l"))
    arrow(ax, c(s, "r"), c(st, "l"))
    save(fig, "fig6_1_evaluation_pipeline")


if __name__ == "__main__":
    architecture()
    workflow()
    coach()
    qaps()
    pipeline()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
