from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalQualityDecision:
    is_adequate: bool
    reason: str
    top_score: float | None = None


class RetrievalQualityService:
    def __init__(self, *, min_score: float = 0.75) -> None:
        self.min_score = min_score

    def evaluate(self, *, question: str, hits: list[dict]) -> RetrievalQualityDecision:
        if not hits:
            return RetrievalQualityDecision(is_adequate=False, reason="retrieval_quality_insufficient", top_score=None)

        best_score = float(max((float(hit.get("score", 0.0)) for hit in hits), default=0.0))
        if best_score >= self.min_score:
            return RetrievalQualityDecision(is_adequate=True, reason="retrieval_quality_adequate", top_score=best_score)
        return RetrievalQualityDecision(is_adequate=False, reason="retrieval_quality_insufficient", top_score=best_score)
