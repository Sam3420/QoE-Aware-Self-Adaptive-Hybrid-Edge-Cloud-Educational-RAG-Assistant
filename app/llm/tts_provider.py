from typing import Protocol

from app.services.errors import TTSProviderError


class TTSProvider(Protocol):
    def synthesize(self, text: str) -> bytes:
        ...


class PassthroughTTSProvider:
    """Minimal provider boundary for environments without a real TTS backend."""

    provider_name = "passthrough"

    def __init__(self, *, provider_name: str = "passthrough", enabled: bool = True) -> None:
        self.provider_name = provider_name
        self.enabled = enabled

    def synthesize(self, text: str) -> bytes:
        if not self.enabled:
            raise TTSProviderError("Text-to-speech provider is disabled in the current configuration.")
        if not text or not text.strip():
            raise TTSProviderError("No text was provided for synthesis.")
        return text.encode("utf-8")
