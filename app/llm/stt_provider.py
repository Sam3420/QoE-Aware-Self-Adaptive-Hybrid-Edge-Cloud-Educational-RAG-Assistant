from typing import Protocol

from app.services.errors import STTProviderError


class STTProvider(Protocol):
    def transcribe(self, audio_bytes: bytes, *, mime_type: str | None = None) -> str:
        ...


class PassthroughSTTProvider:
    """Minimal provider boundary for environments without a real STT backend."""

    provider_name = "passthrough"

    def __init__(self, *, provider_name: str = "passthrough", enabled: bool = True) -> None:
        self.provider_name = provider_name
        self.enabled = enabled

    def transcribe(self, audio_bytes: bytes, *, mime_type: str | None = None) -> str:
        if not self.enabled:
            raise STTProviderError("Speech-to-text provider is disabled in the current configuration.")
        if not audio_bytes:
            raise STTProviderError("No audio data was supplied for transcription.")
        try:
            return audio_bytes.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise STTProviderError("Audio payload could not be decoded as text for the passthrough STT provider.") from exc
