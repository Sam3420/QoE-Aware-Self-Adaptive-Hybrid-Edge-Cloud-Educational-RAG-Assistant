class AssistantError(RuntimeError):
    """Base class for assistant workflow failures."""


class LearningSessionNotFoundError(AssistantError):
    """Raised when a text question targets an unknown learning session."""


class LearnerNotFoundError(AssistantError):
    """Raised when a learner record is missing."""


class LearningSessionOwnershipError(AssistantError):
    """Raised when a session does not belong to the learner being targeted."""


class RuntimeConfigurationNotFoundError(AssistantError):
    """Raised when a requested runtime configuration does not exist."""


class ResourceNotFoundError(AssistantError):
    """Raised when a requested educational resource cannot be found."""


class YouTubeProviderConfigurationError(AssistantError):
    """Raised when YouTube configuration is missing or invalid."""


class YouTubeProviderError(AssistantError):
    """Raised when the YouTube provider fails to return usable results."""


class NoRecommendationsFoundError(AssistantError):
    """Raised when no usable educational recommendations are available."""


class KnowledgePreparationError(AssistantError):
    """Raised when transcript preparation or indexing cannot be completed."""


class KnowledgeRetrievalError(AssistantError):
    """Raised when knowledge retrieval fails."""


class AssistantProviderUnavailableError(AssistantError):
    """Raised after a provider failure has been sanitized and recorded."""

    def __init__(self, message: str, *, interaction_id: str | None = None) -> None:
        super().__init__(message)
        self.interaction_id = interaction_id


class STTProviderError(AssistantError):
    """Raised when speech-to-text transcription fails."""


class TTSProviderError(AssistantError):
    """Raised when text-to-speech synthesis fails."""
