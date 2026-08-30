from __future__ import annotations

from typing import Protocol


class TranscriptProvider(Protocol):
    def get_transcript_for_resource(self, *, resource_id: str, resource: object) -> str:
        ...
