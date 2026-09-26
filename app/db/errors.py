class ImmutableRuntimeConfigurationError(RuntimeError):
    """Raised when a referenced runtime configuration snapshot is modified."""


class ReferencedRecordNotFoundError(LookupError):
    """Raised when a new row references a learner, session, or policy that does not exist."""
