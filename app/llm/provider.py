from typing import Protocol

from app.llm.types import LLMGenerateRequest, LLMGenerateResult


class LLMProvider(Protocol):
    def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        ...
