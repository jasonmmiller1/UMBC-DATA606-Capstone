"""Application package for RMF Assistant."""

from __future__ import annotations

from pathlib import Path


def _load_local_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except Exception:
        print(
            "Warning: .env file detected but python-dotenv is not installed. "
            "Install with: pip install python-dotenv"
        )
        return
    load_dotenv(dotenv_path=env_path, override=False)


_load_local_env()
