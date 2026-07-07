"""Tiny on-disk JSON cache keyed by a content hash.

Vision calls are the expensive part of the pipeline. We key each cached
result on the SHA-256 of the exact page image bytes (plus a namespace so the
same page can hold results for different prompt versions). Re-running a scan
is then free for unchanged pages.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


def content_hash(data: bytes) -> str:
    """SHA-256 hex digest of arbitrary bytes."""
    return hashlib.sha256(data).hexdigest()


class JSONCache:
    """Namespaced JSON blob store under a directory."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str) -> Path:
        ns_dir = self.root / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        return ns_dir / f"{key}.json"

    def get(self, namespace: str, key: str) -> Optional[Any]:
        path = self._path(namespace, key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, namespace: str, key: str, value: Any) -> None:
        path = self._path(namespace, key)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
