"""A tiny content-addressed parquet cache for downloaded market data.

The cache key is a SHA-256 of the request parameters, so a given
(symbol, start, end, interval) always maps to the same file and re-downloads are
avoided. Cached frames are regenerable, so the directory is git-ignored.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path("data/cache")


def request_key(**params: object) -> str:
    """Deterministic hash of request parameters."""
    payload = "|".join(f"{k}={params[k]!r}" for k in sorted(params))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_path(key: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return cache_dir / f"{key}.parquet"


def read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def write_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
