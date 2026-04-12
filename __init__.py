"""Content Moderation Environment package exports."""

from typing import Any

from .models import (
    ModerationAction,
    ModerationObservation,
    ModerationState,
)

__all__ = [
    "ContentModerationEnv",
    "ModerationAction",
    "ModerationObservation",
    "ModerationState",
]


def __getattr__(name: str) -> Any:
    if name == "ContentModerationEnv":
        from .client import ContentModerationEnv as _ContentModerationEnv

        return _ContentModerationEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
