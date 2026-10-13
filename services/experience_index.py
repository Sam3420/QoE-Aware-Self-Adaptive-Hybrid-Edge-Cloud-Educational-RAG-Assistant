"""FAISS index of experience state vectors (similarity search only; the database holds the records).

An exact L2 index (IndexFlatL2) wrapped in IndexIDMap2, so every vector carries its `faiss_id` from the
`experience_vectors` table. The table is the source of truth: `sync()` rebuilds the index whenever its
ids differ from the table (missed add, deleted rows, or a stale file on disk).
"""
import json
import logging
import threading
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissExperienceIndex:
    """Thread-safe exact nearest-neighbour index keyed by faiss_id, optionally persisted to disk."""

    def __init__(self, dim: int, *, directory: Path | None = None, version: str = "v1") -> None:
        self.dim = dim
        self.version = version
        self.directory = Path(directory) if directory else None
        self._lock = threading.Lock()
        self._index = self._empty()
        if self.directory:
            self._load()

    def _empty(self) -> faiss.IndexIDMap2:
        return faiss.IndexIDMap2(faiss.IndexFlatL2(self.dim))

    @property
    def _paths(self) -> tuple[Path, Path]:
        base = self.directory / f"experiences-{self.version}"
        return base.with_suffix(".faiss"), base.with_suffix(".json")

    def _load(self) -> None:
        index_path, meta_path = self._paths
        if not index_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            index = faiss.read_index(str(index_path))
            if meta.get("dim") == self.dim and meta.get("version") == self.version and index.d == self.dim:
                self._index = index
        except Exception as exc:  # a corrupt file is rebuilt from the database by sync()
            logger.warning("experience index unreadable, will rebuild: %s", exc)

    def _save(self) -> None:
        if not self.directory:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        index_path, meta_path = self._paths
        faiss.write_index(self._index, str(index_path))
        meta_path.write_text(json.dumps({"dim": self.dim, "version": self.version, "count": self._index.ntotal}),
                             encoding="utf-8")

    @staticmethod
    def _matrix(vectors: list[list[float]]) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(vectors, dtype="float32"))

    def ids(self) -> set[int]:
        with self._lock:
            return {int(i) for i in faiss.vector_to_array(self._index.id_map)} if self._index.ntotal else set()

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    def add(self, faiss_id: int, vector: list[float]) -> None:
        if len(vector) != self.dim:
            raise ValueError(f"vector has {len(vector)} dims, index expects {self.dim}")
        with self._lock:
            self._index.add_with_ids(self._matrix([vector]), np.asarray([faiss_id], dtype="int64"))
            self._save()

    def rebuild(self, items: list[tuple[int, list[float]]]) -> None:
        with self._lock:
            self._index = self._empty()
            if items:
                self._index.add_with_ids(self._matrix([v for _, v in items]),
                                         np.asarray([i for i, _ in items], dtype="int64"))
            self._save()

    def sync(self, items: list[tuple[int, list[float]]]) -> bool:
        """Rebuild if the index ids differ from the database's; returns True when it rebuilt."""
        if self.ids() == {i for i, _ in items}:
            return False
        logger.info("experience index out of sync (%d in index, %d in db): rebuilding", self.ntotal, len(items))
        self.rebuild(items)
        return True

    def search(self, vector: list[float], k: int) -> list[tuple[int, float]]:
        """The k nearest (faiss_id, Euclidean distance), closest first."""
        with self._lock:
            if self._index.ntotal == 0 or k <= 0:
                return []
            distances, ids = self._index.search(self._matrix([vector]), min(k, self._index.ntotal))
        return [(int(i), float(np.sqrt(max(d, 0.0)))) for d, i in zip(distances[0], ids[0]) if i != -1]
