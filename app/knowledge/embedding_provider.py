from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    # Both document and query embeddings must come from the same implementation/model.
    model_name: str

    def embed_text(self, text: str) -> list[float]:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


class SentenceTransformerEmbeddingProvider:
    def __init__(self, *, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "sentence-transformers is required for embedding generation."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        # Used for one incoming question during retrieval.
        return self._load_model().encode(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Used once per prepared transcript to create vectors for all chunks.
        return self._load_model().encode(texts).tolist()
