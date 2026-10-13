"""Small file cache with per-namespace TTLs, used for API responses and transcripts.

Entries older than their TTL are ignored and deleted on read, and a TTL of 0 disables storage.
Settings cap every TTL at 30 days (YouTube API Services policies on stored API data).
"""
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JsonFileCache:
    """Key → JSON value under base_dir/<namespace>/<sha1(key)>.json, with hit/miss counters."""

    def __init__(self, base_dir: Path, ttls: dict[str, float]) -> None:
        self.base_dir = Path(base_dir)
        self.ttls = ttls
        self.hits = 0
        self.misses = 0

    def _path(self, namespace: str, key: str) -> Path:
        return self.base_dir / namespace / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.json"

    def get(self, namespace: str, key: str) -> Any | None:
        """Return the cached value, or None if missing, expired or unreadable."""
        ttl = self.ttls.get(namespace, 0)
        path = self._path(namespace, key)
        if ttl <= 0 or not path.exists():
            self.misses += 1
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - float(entry["stored_at"]) > ttl:
                path.unlink(missing_ok=True)  # expired: delete rather than keep stale data
                self.misses += 1
                return None
            self.hits += 1
            return entry["value"]
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("cache entry unreadable (%s): %s", path.name, exc)
            path.unlink(missing_ok=True)
            self.misses += 1
            return None

    def set(self, namespace: str, key: str, value: Any) -> None:
        """Store a JSON-serialisable value (no-op when the namespace TTL is 0)."""
        if self.ttls.get(namespace, 0) <= 0:
            return
        path = self._path(namespace, key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"stored_at": time.time(), "key": key, "value": value}, default=str),
                            encoding="utf-8")
        except OSError as exc:  # a cache failure must never fail the request
            logger.warning("cache write failed (%s): %s", path.name, exc)

    def purge_expired(self) -> int:
        """Delete every expired entry; return how many were removed."""
        removed = 0
        for namespace, ttl in self.ttls.items():
            folder = self.base_dir / namespace
            if not folder.is_dir():
                continue
            for path in folder.glob("*.json"):
                try:
                    stored_at = float(json.loads(path.read_text(encoding="utf-8"))["stored_at"])
                except (OSError, ValueError, KeyError):
                    stored_at = 0.0
                if ttl <= 0 or time.time() - stored_at > ttl:
                    path.unlink(missing_ok=True)
                    removed += 1
        return removed
