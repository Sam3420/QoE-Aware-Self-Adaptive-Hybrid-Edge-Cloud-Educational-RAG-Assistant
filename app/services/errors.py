class AssistantError(RuntimeError):
    """Base class for assistant workflow failures."""


class LearningSessionNotFoundError(AssistantError):
    """Raised when a text question targets an unknown learning session."""


class RuntimeConfigurationNotFoundError(AssistantError):
    """Raised when a requested runtime configuration does not exist."""


class AssistantProviderUnavailableError(AssistantError):
    """Raised after a provider failure has been sanitized and recorded."""

    def __init__(self, message: str, *, interaction_id: str | None = None) -> None:
        super().__init__(message)
        self.interaction_id = interaction_id
