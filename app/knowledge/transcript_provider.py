from __future__ import annotations

from typing import Protocol


class TranscriptProvider(Protocol):
    # Transcript acquisition is an interface so YouTube or another provider can be swapped in.
    def get_transcript_for_resource(self, *, resource_id: str, resource: object) -> str:
        ...
