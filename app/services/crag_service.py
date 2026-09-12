from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalQualityDecision:
    # This decision is passed to the assistant before retrieved text enters the prompt.
    is_adequate: bool
    reason: str
    top_score: float | None = None


class RetrievalQualityService:
    def __init__(self, *, min_score: float = 0.75) -> None:
        # The threshold is configurable so experiments can tune the CRAG gate.
        self.min_score = min_score

    def evaluate(self, *, question: str, hits: list[dict]) -> RetrievalQualityDecision:
        # Empty retrieval is always inadequate because there is no evidence to ground on.
        if not hits:
            return RetrievalQualityDecision(is_adequate=False, reason="retrieval_quality_insufficient", top_score=None)

        # Use the strongest retrieved result as the simple quality signal for this phase.
        best_score = float(max((float(hit.get("score", 0.0)) for hit in hits), default=0.0))
        if best_score >= self.min_score:
            return RetrievalQualityDecision(is_adequate=True, reason="retrieval_quality_adequate", top_score=best_score)
        return RetrievalQualityDecision(is_adequate=False, reason="retrieval_quality_insufficient", top_score=best_score)
