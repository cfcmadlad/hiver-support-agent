import hashlib
import json
from pathlib import Path
from typing import Any


class CacheCorruptionError(Exception):
    pass


class ResponseCache:
    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> str | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise CacheCorruptionError(f"corrupted cache entry at {path}") from e
        value = payload.get("value")
        if not isinstance(value, str):
            raise CacheCorruptionError(f"cache entry at {path} has no string value")
        return value

    def set(self, key: str, value: str) -> None:
        path = self._path_for(key)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps({"value": value}), encoding="utf-8")
        tmp_path.replace(path)

    def _path_for(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"


def hash_request(model: str, system: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps({"model": model, "system": system, "payload": payload}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
