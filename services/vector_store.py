"""Embeddings and per-topic FAISS indexes (vectors in .faiss, chunk text in a .json sidecar)."""
import hashlib
import json
import re
from pathlib import Path
from typing import Protocol

import faiss
import httpx
import numpy as np
from pydantic import BaseModel, Field

from telemetry import tracing


class Chunk(BaseModel):
    """One piece of transcript text stored in an index."""

    chunk_id: str
    text: str = Field(min_length=1)
    source: str = ""


class RetrievedChunk(Chunk):
    """A chunk returned by search, with its cosine similarity to the query."""

    score: float = Field(ge=0.0, le=1.0)


class IndexMeta(BaseModel):
    """Sidecar metadata; row i of the FAISS index is chunks[i]."""

    resource_id: str
    embedding_model: str
    dim: int = Field(gt=0)
    chunks: list[Chunk]


class IndexNotFoundError(LookupError):
    """Raised when no index has been built for a resource."""


class EmbeddingError(RuntimeError):
    """Raised when the embedding backend cannot embed text."""


class Embedder(Protocol):
    """Turns text into L2-normalised float32 vectors."""

    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so inner product equals cosine similarity."""
    matrix = np.asarray(matrix, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class OllamaEmbedder:
    """Embeddings from a local Ollama model (default nomic-embed-text)."""

    def __init__(self, *, model: str, base_url: str, timeout_s: float = 120.0) -> None:
        self.model = model
        self.name = f"ollama:{model}"
        self._url = f"{base_url.rstrip('/')}/api/embed"
        self._http = httpx.Client(timeout=timeout_s)
        # nomic-embed-text is trained with task prefixes; other models get none.
        self._nomic = "nomic" in model

    def _embed(self, texts: list[str], operation: str) -> np.ndarray:
        """Call Ollama's /api/embed for a batch of texts."""
        with tracing.observation(operation, as_type="embedding", model=self.model,
                                 input={"texts": len(texts), "first": tracing.preview(texts[0], 200) if texts else None},
                                 metadata={"provider": "ollama"}) as obs:
            try:
                resp = self._http.post(self._url, json={"model": self.model, "input": texts})
                resp.raise_for_status()
                data = resp.json()
                vectors = _normalise(np.array(data["embeddings"]))
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                raise EmbeddingError(f"Ollama embedding failed ({self.model}): {exc}") from exc
            tokens = data.get("prompt_eval_count")
            obs.update(output={"vectors": int(vectors.shape[0]), "dim": int(vectors.shape[1])},
                       usage_details={"input": int(tokens)} if isinstance(tokens, int) else None)
            return vectors

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed transcript chunks."""
        return self._embed([f"search_document: {t}" if self._nomic else t for t in texts], "embed-documents")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one learner question."""
        return self._embed([f"search_query: {text}" if self._nomic else text], "embed-query")[0]


class HashingEmbedder:
    """Model-free bag-of-words hashing embedder, for offline runs and tests."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing:{dim}"

    def _vector(self, text: str) -> np.ndarray:
        """Hash each lowercase token into one of `dim` buckets."""
        vec = np.zeros(self.dim, dtype="float32")
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed transcript chunks."""
        return _normalise(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one learner question."""
        return _normalise(self._vector(text)[None, :])[0]


class FaissVectorStore:
    """Builds and searches one exact inner-product FAISS index per resource (topic/video)."""

    def __init__(self, *, base_dir: Path, embedder: Embedder) -> None:
        self.base_dir = Path(base_dir)
        self.embedder = embedder
        self._cache: dict[str, tuple[faiss.Index, IndexMeta]] = {}

    def _paths(self, resource_id: str) -> tuple[Path, Path]:
        """Return the (index, sidecar) file paths for a resource."""
        return self.base_dir / f"{resource_id}.faiss", self.base_dir / f"{resource_id}.json"

    def exists(self, resource_id: str) -> bool:
        """Report whether an index has been built for the resource."""
        return all(p.exists() for p in self._paths(resource_id))

    def build(self, resource_id: str, chunks: list[Chunk]) -> IndexMeta:
        """Embed chunks and write the index plus its sidecar, replacing any existing one."""
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks.")
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        meta = IndexMeta(resource_id=resource_id, embedding_model=self.embedder.name,
                         dim=vectors.shape[1], chunks=chunks)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        index_path, meta_path = self._paths(resource_id)
        faiss.write_index(index, str(index_path))
        meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
        self._cache[resource_id] = (index, meta)
        return meta

    def _load(self, resource_id: str) -> tuple[faiss.Index, IndexMeta]:
        """Load (and cache) an index, checking that it was built with the same embedder."""
        if resource_id not in self._cache:
            if not self.exists(resource_id):
                raise IndexNotFoundError(f"No vector index for resource '{resource_id}' in {self.base_dir}.")
            index_path, meta_path = self._paths(resource_id)
            meta = IndexMeta.model_validate(json.loads(meta_path.read_text(encoding="utf-8")))
            if meta.embedding_model != self.embedder.name:
                raise EmbeddingError(
                    f"Index '{resource_id}' was built with {meta.embedding_model}, "
                    f"but the query embedder is {self.embedder.name}; rebuild the index."
                )
            self._cache[resource_id] = (faiss.read_index(str(index_path)), meta)
        return self._cache[resource_id]

    def score_chunks(self, resource_id: str, query: str, chunk_ids: list[str]) -> dict[str, float]:
        """Return the cosine similarity between a query and specific stored chunks."""
        index, meta = self._load(resource_id)
        rows = {c.chunk_id: i for i, c in enumerate(meta.chunks)}
        query_vec = self.embedder.embed_query(query)
        return {
            cid: float(np.clip(index.reconstruct(rows[cid]) @ query_vec, 0.0, 1.0))
            for cid in chunk_ids if cid in rows
        }

    def search(self, resource_id: str, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the top_k chunks by cosine similarity, highest first."""
        index, meta = self._load(resource_id)
        query_vec = self.embedder.embed_query(query)[None, :]
        scores, ids = index.search(query_vec, min(top_k, index.ntotal))
        return [
            RetrievedChunk(**meta.chunks[i].model_dump(), score=float(np.clip(s, 0.0, 1.0)))
            for s, i in zip(scores[0], ids[0])
            if 0 <= i < len(meta.chunks)
        ]
