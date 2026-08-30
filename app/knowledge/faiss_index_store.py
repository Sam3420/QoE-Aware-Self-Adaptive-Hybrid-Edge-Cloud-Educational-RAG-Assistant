from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:  # pragma: no cover
    import faiss
except Exception:  # pragma: no cover
    faiss = None


class FaissIndexStore:
    def __init__(self, *, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def build_index(self, *, resource_id: str, embeddings: list[list[float]], chunk_ids: list[str]) -> dict:
        if faiss is None:
            raise RuntimeError("faiss is required for vector retrieval.")
        if not embeddings:
            raise ValueError("No embeddings available to build the FAISS index.")
        matrix = np.asarray(embeddings, dtype="float32")
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        index_path = self.base_dir / f"{resource_id}.faiss"
        faiss.write_index(index, str(index_path))
        metadata_path = self.base_dir / f"{resource_id}.json"
        metadata_path.write_text(
            json.dumps({"resource_id": resource_id, "chunk_ids": chunk_ids}, indent=2),
            encoding="utf-8",
        )
        return {"resource_id": resource_id, "path": str(index_path), "chunk_count": len(chunk_ids)}

    def search(self, *, resource_id: str, query_vector: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        if faiss is None:
            raise RuntimeError("faiss is required for vector retrieval.")
        index_path = self.base_dir / f"{resource_id}.faiss"
        if not index_path.exists():
            return []
        index = faiss.read_index(str(index_path))
        metadata_path = self.base_dir / f"{resource_id}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {"chunk_ids": []}
        query = np.asarray([query_vector], dtype="float32")
        score_matrix, indices = index.search(query, min(top_k, index.ntotal))
        ranked = []
        for score, idx in zip(score_matrix[0], indices[0]):
            if idx < 0 or idx >= len(metadata.get("chunk_ids", [])):
                continue
            chunk_id = metadata["chunk_ids"][int(idx)]
            ranked.append((chunk_id, float(score)))
        return ranked
