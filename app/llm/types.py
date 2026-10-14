from pydantic import BaseModel, Field


class LLMGenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    system_prompt: str | None = None
    model_id: str
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0)
    # Optional JSON schema the reply must follow; providers that can enforce one (Ollama) do.
    json_schema: dict | None = None


class LLMGenerateResult(BaseModel):
    text: str
    model_provider: str
    model_name: str


class LLMProviderError(RuntimeError):
    """Raised when a provider cannot produce a safe text response."""
