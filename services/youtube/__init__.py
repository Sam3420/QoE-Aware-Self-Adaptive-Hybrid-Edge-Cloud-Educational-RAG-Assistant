"""YouTube educational video recommendations.

Two independent sources: the official YouTube Data API v3 (search.list, videos.list) for search and
metadata, and the unofficial youtube-transcript-api behind a replaceable adapter for transcripts.
The captions.download endpoint is deliberately not used: it only serves videos the authenticated
user can manage, so it cannot provide transcripts for arbitrary public videos.
"""
