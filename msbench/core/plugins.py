"""Shared plugin discovery with isolated, inspectable import failures."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points
from collections.abc import Iterable


def discover_plugins(modules: Iterable[str], group: str) -> dict[str, str]:
    errors = {}
    for module in modules:
        try:
            import_module(module)
        except Exception as exc:
            errors[module] = f"{type(exc).__name__}: {exc}"
    try:
        plugins = entry_points(group=group)
    except Exception as exc:
        errors[group] = f"{type(exc).__name__}: {exc}"
        return errors
    for plugin in plugins:
        try:
            # Module entry points register through decorators; class entry
            # points are handled by their consumer's register API.
            plugin.load()
        except Exception as exc:
            errors[f"{group}:{plugin.name}"] = f"{type(exc).__name__}: {exc}"
    return errors
