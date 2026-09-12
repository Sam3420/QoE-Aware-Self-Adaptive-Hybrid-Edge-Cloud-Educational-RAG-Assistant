from __future__ import annotations

import httpx

from app.core.config import Settings
from app.services.errors import YouTubeProviderConfigurationError, YouTubeProviderError

'''
calls the YouTube Data API.

It returns metadata such as:

YouTube video ID
title
description
URL
channel
language
publication date
It does not return transcripts.
'''

class YouTubeDataAPIProvider:
    provider_name = "youtube"

    def __init__(self, *, settings: Settings) -> None:
        # The API key is kept private and is used only for outbound YouTube requests.
        self.settings = settings
        api_key = settings.youtube_api_key.get_secret_value() if settings.youtube_api_key else ""
        if not api_key:
            raise YouTubeProviderConfigurationError("YouTube API key is not configured.")
        self._api_key = api_key

    def search_educational_videos(
        self,
        *,
        query: str,
        language: str | None,
        max_results: int,
    ) -> list[dict]:
        # YouTube Data API returns searchable video metadata, not transcript text.
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "videoEmbeddable": "true",
            "safeSearch": "moderate",
            "maxResults": max_results,
            "key": self._api_key,
        }
        if language:
            params["relevanceLanguage"] = language

        try:
            response = httpx.get(
                self.settings.youtube_api_base_url,
                params=params,
                timeout=self.settings.youtube_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise YouTubeProviderError("YouTube API request timed out.") from exc
        except httpx.HTTPError as exc:
            raise YouTubeProviderError("YouTube API request failed.") from exc

        payload = response.json()
        items = payload.get("items", [])
        if not isinstance(items, list):
            raise YouTubeProviderError("YouTube API returned an invalid response.")

        # Normalize provider-specific JSON into the application’s candidate shape.
        normalized: list[dict] = []
        for item in items:
            snippet = item.get("snippet") or {}
            video_id = item.get("id", {}).get("videoId") if isinstance(item.get("id"), dict) else None
            if not video_id:
                continue
            normalized.append(
                {
                    "provider": self.provider_name,
                    "external_id": video_id,
                    "title": snippet.get("title") or "Untitled video",
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "channel_title": snippet.get("channelTitle"),
                    "description": snippet.get("description") or "",
                    "language": snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage"),
                    "published_at": snippet.get("publishedAt"),
                    "topic": query,
                }
            )
        return normalized
