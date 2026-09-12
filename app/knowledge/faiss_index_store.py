from __future__ import annotations

# JSON stores the mapping between FAISS vector positions and database chunk IDs.
import json
# Path supports both string and pathlib storage configuration values.
from pathlib import Path

# NumPy converts Python lists into FAISS-compatible numeric matrices.
import numpy as np

# FAISS is optional at import time so the application can report a clear runtime error.
try:  # pragma: no cover
    import faiss
except Exception:  # pragma: no cover
    faiss = None


class FaissIndexStore:
    # This class owns on-disk vector indexes, not the complete educational text.
    def __init__(self, *, base_dir: str | Path):
        # Keep vector files separate from relational application data.
        # Normalize the configured location to a Path for portable file operations.
        self.base_dir = Path(base_dir)
        # Create the storage directory before any index is written.
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def build_index(self, *, resource_id: str, embeddings: list[list[float]], chunk_ids: list[str]) -> dict:
        # Fail with a specific message if the optional vector dependency is unavailable.
        if faiss is None:
            raise RuntimeError("faiss is required for vector retrieval.")
        # An index without vectors cannot answer retrieval queries.
        if not embeddings:
            raise ValueError("No embeddings available to build the FAISS index.")
        # Inner-product search ranks chunks by vector similarity.
        # float32 is the numeric format expected by FAISS.
        matrix = np.asarray(embeddings, dtype="float32")
        # IndexFlatIP provides exact inner-product search for this prototype.
        index = faiss.IndexFlatIP(matrix.shape[1])
        # Add one vector per transcript chunk.
        index.add(matrix)
        # Store a FAISS file and a sidecar mapping from vector position to chunk ID.
        index_path = self.base_dir / f"{resource_id}.faiss"
        # Persist the binary vector index for later retrieval requests.
        faiss.write_index(index, str(index_path))
        # Store the ID mapping separately because FAISS does not know the database records.
        metadata_path = self.base_dir / f"{resource_id}.json"
        metadata_path.write_text(
            json.dumps({"resource_id": resource_id, "chunk_ids": chunk_ids}, indent=2),
            encoding="utf-8",
        )
        # Return metadata needed by the knowledge preparation service.
        return {"resource_id": resource_id, "path": str(index_path), "chunk_count": len(chunk_ids)}

    def search(self, *, resource_id: str, query_vector: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        # Retrieval also requires FAISS to be installed in the active environment.
        if faiss is None:
            raise RuntimeError("faiss is required for vector retrieval.")
        # Each resource has its own index file.
        index_path = self.base_dir / f"{resource_id}.faiss"
        if not index_path.exists():
            # No index means preparation has not completed for this resource.
            return []
        # Load the persisted vector index and its chunk-ID mapping.
        index = faiss.read_index(str(index_path))
        metadata_path = self.base_dir / f"{resource_id}.json"
        # Missing sidecar metadata is treated as an empty mapping rather than guessed.
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {"chunk_ids": []}
        # FAISS expects a two-dimensional float32 query matrix.
        query = np.asarray([query_vector], dtype="float32")
        # Never ask FAISS for more results than the index contains.
        score_matrix, indices = index.search(query, min(top_k, index.ntotal))
        # Convert FAISS positions into application-level chunk IDs and scores.
        ranked = []
        for score, idx in zip(score_matrix[0], indices[0]):
            # Ignore invalid positions from an inconsistent or stale index mapping.
            if idx < 0 or idx >= len(metadata.get("chunk_ids", [])):
                continue
            # Use the integer vector position to obtain the original database chunk ID.
            chunk_id = metadata["chunk_ids"][int(idx)]
            # Return the pair expected by KnowledgeRetrievalService.
            ranked.append((chunk_id, float(score)))
        # FAISS already returns results in descending similarity order.
        return ranked
