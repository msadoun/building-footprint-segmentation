"""Helpers so scripts can use an in-file CONFIG plus optional CLI overrides."""

from __future__ import annotations

from typing import Any


def apply_cli_overrides(config: dict[str, Any], args: Any) -> dict[str, Any]:
    """
    Return a copy of CONFIG with any non-None argparse values applied.

    Argparse dest names should match CONFIG keys (use dest=... when needed).
    """
    settings = dict(config)
    for key in list(settings.keys()):
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                settings[key] = value
    return settings
