"""Build a topic's FAISS index from a transcript text file.

Usage:
  .venv\\Scripts\\python -m knowledge_prep.build_index photosynthesis
  .venv\\Scripts\\python -m knowledge_prep.build_index my_topic --file path\\to\\transcript.txt
"""
import argparse
import re
from pathlib import Path

from agent.deps import make_embedder
from config import settings
from services.vector_store import Chunk, FaissVectorStore, IndexMeta


def chunk_text(text: str, *, resource_id: str, size: int, overlap: int, source: str = "") -> list[Chunk]:
    """Split text into overlapping word windows, snapping ends to sentence boundaries where possible."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        # Extend to the end of the sentence (up to 20 extra words), so chunks don't cut mid-thought.
        while end < len(words) and end < start + size + 20 and not words[end - 1].endswith((".", "?", "!")):
            end += 1
        chunks.append(Chunk(chunk_id=f"{resource_id}-{len(chunks):04d}", text=" ".join(words[start:end]),
                            source=source))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_topic_index(resource_id: str, transcript_path: Path) -> IndexMeta:
    """Chunk a transcript, embed it with the configured embedder, and write the FAISS index."""
    text = transcript_path.read_text(encoding="utf-8")
    chunks = chunk_text(text, resource_id=resource_id, size=settings.chunk_size_words,
                        overlap=settings.chunk_overlap_words, source=transcript_path.name)
    store = FaissVectorStore(base_dir=settings.vector_index_dir, embedder=make_embedder())
    return store.build(resource_id, chunks)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build a FAISS index for one topic.")
    parser.add_argument("resource_id", help="topic/video id, e.g. photosynthesis")
    parser.add_argument("--file", type=Path, help="transcript .txt (default: knowledge_prep/samples/<id>.txt)")
    args = parser.parse_args()
    path = args.file or settings.knowledge_samples_dir / f"{args.resource_id}.txt"
    meta = build_topic_index(args.resource_id, path)
    print(f"Indexed {len(meta.chunks)} chunks for '{meta.resource_id}' with {meta.embedding_model} "
          f"(dim {meta.dim}) into {settings.vector_index_dir}")


if __name__ == "__main__":
    main()
