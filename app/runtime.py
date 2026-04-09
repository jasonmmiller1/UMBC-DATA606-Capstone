from __future__ import annotations

import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    if value.strip():
        return value.strip()
    return default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "")
    if not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name, "")
    if not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name, "")
    if not value.strip():
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


DATA_DIR = resolve_repo_path(_env_str("APP_DATA_DIR", "data"))
UPLOAD_PDF_DIR = resolve_repo_path(_env_str("UPLOAD_PDF_DIR", str(DATA_DIR / "uploads_pdf")))
UPLOAD_MD_DIR = resolve_repo_path(_env_str("UPLOAD_MD_DIR", str(DATA_DIR / "uploads_md")))
CHUNKS_PATH = resolve_repo_path(_env_str("CHUNKS_PATH", str(DATA_DIR / "index" / "chunks.parquet")))
BM25_INDEX_PATH = resolve_repo_path(_env_str("BM25_INDEX_PATH", str(DATA_DIR / "bm25_index" / "bm25_index.pkl")))
BM25_DIR = BM25_INDEX_PATH.parent

QDRANT_URL = _env_str("QDRANT_URL", "")
QDRANT_API_KEY = _env_str("QDRANT_API_KEY", "")
QDRANT_HOST = _env_str("QDRANT_HOST", "localhost")
QDRANT_PORT = env_int("QDRANT_PORT", 6333)
QDRANT_TIMEOUT = env_float("QDRANT_TIMEOUT", 0.75)
QDRANT_STATUS_ENABLED = env_bool("QDRANT_STATUS_ENABLED", False)
QDRANT_COLLECTION = _env_str("QDRANT_COLLECTION", "rmf_chunks")

PORT = env_int("PORT", env_int("STREAMLIT_SERVER_PORT", 8501))


def qdrant_connection_kwargs(*, timeout: float | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "timeout": float(QDRANT_TIMEOUT if timeout is None else timeout),
    }
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    if QDRANT_URL:
        kwargs["url"] = QDRANT_URL
    else:
        kwargs["host"] = QDRANT_HOST
        kwargs["port"] = QDRANT_PORT
    return kwargs


def create_qdrant_client(*, timeout: float | None = None):
    try:
        from qdrant_client import QdrantClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "qdrant_client dependency is missing. Install requirements and retry."
        ) from exc
    return QdrantClient(**qdrant_connection_kwargs(timeout=timeout))


def qdrant_target_label() -> str:
    if QDRANT_URL:
        return QDRANT_URL
    return f"{QDRANT_HOST}:{QDRANT_PORT}"
