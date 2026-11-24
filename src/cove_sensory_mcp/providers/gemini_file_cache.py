"""Nox fork: remember Gemini File API uploads so a big video is uploaded once.

Entries live in a small JSON file keyed by ``realpath|size|mtime_ns``.  Gemini
keeps uploaded files for 48 hours; entries are trusted for a little less and
dropped on expiry.  The cache holds only remote file names/URIs, never media.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

DEFAULT_CACHE_PATH = "~/nox-state/sensory/files.json"
_LIFETIME = timedelta(hours=46)


@dataclass(frozen=True, slots=True)
class CachedFile:
    name: str
    uri: str
    mime_type: str
    expires_at: datetime


def cache_key(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    return f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"


class GeminiFileCache:
    def __init__(self, path: str | os.PathLike[str], now: Any = None) -> None:
        self.path = Path(os.path.expanduser(str(path)))
        self._now = now or (lambda: datetime.now(UTC))

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with open(lock_path, "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                data = json.loads(self.path.read_text()) if self.path.exists() else {}
                if not isinstance(data, dict):
                    data = {}
            except (OSError, ValueError):
                data = {}
            before = json.dumps(data, sort_keys=True)
            yield data
            if json.dumps(data, sort_keys=True) != before:
                fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".files-")
                with os.fdopen(fd, "w") as handle:
                    json.dump(data, handle, ensure_ascii=False, indent=1, sort_keys=True)
                os.replace(tmp, self.path)

    def _prune(self, data: dict[str, Any]) -> None:
        now = self._now()
        for key in list(data):
            try:
                expires = datetime.fromisoformat(data[key]["expires_at"])
            except (KeyError, TypeError, ValueError):
                expires = now
            if expires <= now:
                del data[key]

    def get(self, media_path: Path) -> CachedFile | None:
        try:
            key = cache_key(media_path)
        except OSError:
            return None
        with self._locked() as data:
            self._prune(data)
            entry = data.get(key)
            if not isinstance(entry, dict):
                return None
            try:
                return CachedFile(
                    name=str(entry["name"]),
                    uri=str(entry["uri"]),
                    mime_type=str(entry["mime_type"]),
                    expires_at=datetime.fromisoformat(entry["expires_at"]),
                )
            except (KeyError, ValueError):
                data.pop(key, None)
                return None

    def put(self, media_path: Path, *, name: str, uri: str, mime_type: str) -> None:
        key = cache_key(media_path)
        with self._locked() as data:
            self._prune(data)
            data[key] = {
                "name": name,
                "uri": uri,
                "mime_type": mime_type,
                "expires_at": (self._now() + _LIFETIME).isoformat(),
            }

    def forget(self, media_path: Path) -> None:
        try:
            key = cache_key(media_path)
        except OSError:
            return
        with self._locked() as data:
            data.pop(key, None)
