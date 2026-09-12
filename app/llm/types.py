from pydantic import BaseModel, Field


class LLMGenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    system_prompt: str | None = None
    model_id: str
    provider: str | None = None
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0)


class LLMGenerateResult(BaseModel):
    text: str
    model_provider: str
    model_name: str


class LLMProviderError(RuntimeError):
    """Raised when a provider cannot produce a safe text response."""
