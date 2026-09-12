from __future__ import annotations

# Third-party client that retrieves captions/transcripts for public YouTube videos.
from youtube_transcript_api import YouTubeTranscriptApi


class YouTubeTranscriptProvider:
    # This adapter implements the application's TranscriptProvider contract.
    def __init__(self, *, languages: list[str] | None = None) -> None:
        # Prefer configured languages and fall back to English for the demo.
        self.languages = languages or ["en"]
        # Keep one client instance for transcript requests made by this provider.
        self.client = YouTubeTranscriptApi()

    def get_transcript_for_resource(self, *, resource_id: str, resource: object) -> str:
        # Resource metadata stores the YouTube video ID separately from the database resource ID.
        external_resource_id = getattr(resource, "external_resource_id", "")
        if not external_resource_id:
            # Preparation cannot continue when the resource has no video identifier.
            raise ValueError(f"Resource has no YouTube video ID: {resource_id}")

        # Fetch captions in the preferred language from the external transcript service.
        transcript = self.client.fetch(
            external_resource_id,
            languages=self.languages,
        )
        # Convert timestamped caption snippets into one plain text transcript for preprocessing.
        return " ".join(snippet.text for snippet in transcript)