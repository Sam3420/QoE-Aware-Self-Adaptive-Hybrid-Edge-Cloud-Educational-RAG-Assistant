"""Run benchmarks against a copy of data/app.db, a separate experience index and no tracing.

Call isolate() first, before importing config.py or anything that does: the settings read these environment
variables once. The real database, experience memory and Langfuse project are never written by a benchmark.
"""
import os
import shutil
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "evaluation" / "results"
WORK = RESULTS / "_work"  # database copies and scratch indexes (gitignored)

# Experience memory of the adaptive engine: emptied in a copy that starts the engine cold.
EXPERIENCE_TABLES = ("experience_vectors", "experience_transitions", "experiences")


def isolate(run: str, *, fresh_copy: bool = True, empty_experiences: bool = False) -> Path:
    """Point the app at evaluation/results/_work/<run>/app.db (copied from data/app.db) and return that folder."""
    work = WORK / run
    work.mkdir(parents=True, exist_ok=True)
    db = work / "app.db"
    if fresh_copy or not db.exists():
        with sqlite3.connect(REPO / "data" / "app.db") as src, sqlite3.connect(db) as dst:
            src.backup(dst)  # consistent copy even while the app is running
        if empty_experiences:
            with sqlite3.connect(db) as con:
                for table in EXPERIENCE_TABLES:
                    con.execute(f"DELETE FROM {table}")
        shutil.rmtree(work / "experience_index", ignore_errors=True)
    os.environ.update({
        "DATABASE_PATH": str(db),
        "EXPERIENCE_INDEX_DIR": str(work / "experience_index"),
        "CHECKPOINT_DB_PATH": str(work / "checkpoints.db"),
        "WARMUP_ON_START": "false",
        "PREWARM_STT": "false",
        "PREWARM_VIDEO_CACHES": "false",
        "VIDEO_PREFETCH_COUNT": "0",
        "LQ_SOURCE": "off",
    })
    os.chdir(REPO)
    return work


def disable_tracing() -> None:
    """No benchmark traces in the project's Langfuse (they would mix with real sessions)."""
    from telemetry import tracing

    tracing.set_client(None)
