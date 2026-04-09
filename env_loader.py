from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

def load_environment(root: str | Path | None = None) -> str | None:
    """Load .env only when LOAD_ENV_FILE is enabled."""
    project_root = Path(root) if root is not None else Path(__file__).resolve().parent
    should_load = os.getenv("LOAD_ENV_FILE", "").strip().lower() in {"1", "true", "yes", "on"}
    env_path = project_root / ".env"

    if should_load and env_path.exists():
        load_dotenv(env_path, override=False)
        return str(env_path)

    return None
