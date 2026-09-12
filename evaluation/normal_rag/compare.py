"""Run a paired normal-RAG versus CRAG evaluation with full JSON logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.db.base import Base
from app.db.repositories import LearningTraceRepository
from app.knowledge.embedding_provider import SentenceTransformerEmbeddingProvider
from app.knowledge.faiss_index_store import FaissIndexStore
from app.llm.huggingface_provider import HuggingFaceInferenceProvider
from app.llm.types import LLMGenerateRequest
from app.services.crag_service import RetrievalQualityService
from app.services.knowledge_retrieval_service import KnowledgeRetrievalService


def _prompt(question: str, hits: list[dict], *, crag: bool) -> tuple[str, str]:
    context = "\n".join(f"- {hit['content']}" for hit in hits)
    instruction = "Use only the retrieved context." if crag else "Answer using the retrieved context."
    return f"{instruction} Say when the context is insufficient.\n{context}", question


def _score(*, success: bool, top_score: float | None, latency_ms: int, answer: str) -> float:
    retrieval = max(0.0, min(1.0, top_score or 0.0))
    latency = max(0.0, min(1.0, 1.0 - latency_ms / 10000.0))
    length = max(0.0, min(1.0, len(answer) / 500.0))
    return round(100 * (0.4 * retrieval + 0.3 * float(success) + 0.2 * latency + 0.1 * length), 2)


def compare(questions_path: Path, *, resource_id: str, settings: Settings, output_dir: Path) -> dict[str, object]:
    engine = create_engine(settings.resolved_database_url, future=True)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    with session_factory() as session:
        repository = LearningTraceRepository(session)
        retrieval = KnowledgeRetrievalService(
            repository=repository,
            embedding_provider=SentenceTransformerEmbeddingProvider(model_name=settings.knowledge_embedding_model),
            vector_index_store=FaissIndexStore(base_dir=settings.faiss_storage_path),
            settings=settings,
        )
        token = settings.hf_token.get_secret_value() if settings.hf_token else ""
        provider = HuggingFaceInferenceProvider(token=token, inference_url=settings.hf_inference_url, timeout_seconds=settings.llm_timeout_seconds)
        gate = RetrievalQualityService(min_score=settings.crag_quality_min_score)
        records: list[dict[str, object]] = []
        questions = json.loads(questions_path.read_text(encoding="utf-8"))
        for item in questions:
            question = item["question"]
            normal_hits = retrieval.retrieve_context(resource_id=resource_id, query=question)
            normal = _run(provider, question, normal_hits, settings=settings, crag=False)
            first_hits = retrieval.retrieve_context(resource_id=resource_id, query=question)
            first_decision = gate.evaluate(question=question, hits=first_hits)
            retry_query = None
            retry_hits: list[dict] = []
            final_hits = first_hits
            if not first_decision.is_adequate:
                retry_query = gate.corrective_query(question)
                retry_hits = retrieval.retrieve_context(resource_id=resource_id, query=retry_query)
                final_hits = retry_hits
            final_decision = gate.evaluate(question=retry_query or question, hits=final_hits)
            crag = _run(provider, question, final_hits, settings=settings, crag=True)
            records.append({
                "id": item["id"], "question": question, "difficulty": item.get("difficulty", "unknown"),
                "normal_rag": {"input_hits": normal_hits, **normal},
                "crag": {"initial_hits": first_hits, "initial_decision": first_decision.__dict__, "retry_query": retry_query, "retry_hits": retry_hits, "final_decision": final_decision.__dict__, **crag},
            })
    summary = _summary(records)
    if len(records) != len(questions):
        raise RuntimeError(f"Evaluated {len(records)} of {len(questions)} questions.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    _save_graphs(records, output_dir)
    return {"summary": summary, "records": records}


def _save_graphs(records: list[dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib: pip install matplotlib") from exc

    labels = [record["id"] for record in records]
    normal_qoe = [record["normal_rag"]["qoe_proxy"] for record in records]
    crag_qoe = [record["crag"]["qoe_proxy"] for record in records]
    normal_latency = [record["normal_rag"]["latency_ms"] for record in records]
    crag_latency = [record["crag"]["latency_ms"] for record in records]

    for name, title, ylabel, normal, crag in [
        ("qoe_by_question.png", "QoE Proxy: Normal RAG vs CRAG", "QoE proxy", normal_qoe, crag_qoe),
        ("latency_by_question.png", "Latency: Normal RAG vs CRAG", "Milliseconds", normal_latency, crag_latency),
    ]:
        figure, axis = plt.subplots(figsize=(12, 5))
        positions = list(range(len(labels)))
        axis.plot(positions, normal, marker="o", label="Normal RAG")
        axis.plot(positions, crag, marker="o", label="CRAG")
        axis.set_xticks(positions, labels, rotation=45)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / name, dpi=150)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    means = [sum(normal_qoe) / len(normal_qoe), sum(crag_qoe) / len(crag_qoe)]
    axis.bar(["Normal RAG", "CRAG"], means)
    axis.set_title("Mean QoE Proxy")
    axis.set_ylabel("QoE proxy")
    figure.tight_layout()
    figure.savefig(output_dir / "mean_qoe_comparison.png", dpi=150)
    plt.close(figure)

    difficulties = sorted({record["difficulty"] for record in records})
    normal_by_difficulty = [
        sum(record["normal_rag"]["qoe_proxy"] for record in records if record["difficulty"] == difficulty)
        / sum(record["difficulty"] == difficulty for record in records)
        for difficulty in difficulties
    ]
    crag_by_difficulty = [
        sum(record["crag"]["qoe_proxy"] for record in records if record["difficulty"] == difficulty)
        / sum(record["difficulty"] == difficulty for record in records)
        for difficulty in difficulties
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    positions = list(range(len(difficulties)))
    width = 0.35
    axis.bar([position - width / 2 for position in positions], normal_by_difficulty, width, label="Normal RAG")
    axis.bar([position + width / 2 for position in positions], crag_by_difficulty, width, label="CRAG")
    axis.set_xticks(positions, difficulties)
    axis.set_title("Mean QoE Proxy by Difficulty")
    axis.set_ylabel("QoE proxy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "qoe_by_difficulty.png", dpi=150)
    plt.close(figure)


def _run(provider, question: str, hits: list[dict], *, settings: Settings, crag: bool) -> dict[str, object]:
    system_prompt, prompt = _prompt(question, hits, crag=crag)
    started = perf_counter()
    answer = ""
    error = None
    try:
        result = provider.generate(LLMGenerateRequest(prompt=prompt, system_prompt=system_prompt, model_id=settings.llm_model_id, max_tokens=settings.llm_max_tokens, temperature=settings.llm_temperature))
        answer = result.text
    except Exception as exc:
        error = str(exc)
    latency_ms = max(0, int((perf_counter() - started) * 1000))
    top_score = max((float(hit.get("score", 0.0)) for hit in hits), default=None)
    return {"prompt": prompt, "system_prompt": system_prompt, "answer": answer, "error": error, "success": error is None, "latency_ms": latency_ms, "top_score": top_score, "qoe_proxy": _score(success=error is None, top_score=top_score, latency_ms=latency_ms, answer=answer)}


def _summary(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for name in ("normal_rag", "crag"):
        values = [record[name] for record in records]
        rows.append({"method": name, "count": len(values), "success_rate": round(sum(bool(value["success"]) for value in values) / len(values), 4), "mean_qoe_proxy": round(sum(float(value["qoe_proxy"]) for value in values) / len(values), 2), "mean_latency_ms": round(sum(int(value["latency_ms"]) for value in values) / len(values), 2), "mean_top_score": round(sum(float(value["top_score"] or 0) for value in values) / len(values), 4), "retry_count": sum(bool(record["crag"]["retry_query"]) for record in records) if name == "crag" else 0})
    rows.append({"method": "crag_minus_normal", "count": len(records), "success_rate": round(rows[1]["success_rate"] - rows[0]["success_rate"], 4), "mean_qoe_proxy": round(rows[1]["mean_qoe_proxy"] - rows[0]["mean_qoe_proxy"], 2), "mean_latency_ms": round(rows[1]["mean_latency_ms"] - rows[0]["mean_latency_ms"], 2), "mean_top_score": round(rows[1]["mean_top_score"] - rows[0]["mean_top_score"], 4), "retry_count": rows[1]["retry_count"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--resource-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/normal_rag"))
    args = parser.parse_args()
    result = compare(args.input, resource_id=args.resource_id, settings=Settings(), output_dir=args.output_dir)
    print(json.dumps(result["summary"], indent=2))
    print(f"Detailed logs: {args.output_dir / 'comparison.json'}")
    print(f"Summary CSV: {args.output_dir / 'comparison.csv'}")
    print(f"Graphs: {args.output_dir / 'qoe_by_question.png'}")


if __name__ == "__main__":
    main()
