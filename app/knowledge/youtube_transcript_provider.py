from __future__ import annotations

from youtube_transcript_api import YouTubeTranscriptApi


class YouTubeTranscriptProvider:
    def __init__(self, *, languages: list[str] | None = None) -> None:
        self.languages = languages or ["en"]
        self.client = YouTubeTranscriptApi()

    def get_transcript_for_resource(self, *, resource_id: str, resource: object) -> str:
        external_resource_id = getattr(resource, "external_resource_id", "")
        if not external_resource_id:
            raise ValueError(f"Resource has no YouTube video ID: {resource_id}")

        transcript = self.client.fetch(
            external_resource_id,
            languages=self.languages,
        )
        return " ".join(snippet.text for snippet in transcript)