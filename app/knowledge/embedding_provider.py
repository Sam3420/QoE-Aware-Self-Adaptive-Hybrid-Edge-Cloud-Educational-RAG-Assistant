from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    model_name: str

    def embed_text(self, text: str) -> list[float]:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


class SentenceTransformerEmbeddingProvider:
    def __init__(self, *, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required for embedding generation."
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()
