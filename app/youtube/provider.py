from typing import Protocol


class YouTubeProvider(Protocol):
    def search_educational_videos(
        self,
        *,
        query: str,
        language: str | None,
        max_results: int,
    ) -> list[dict]:
        ...
