"""Load upstream experimental symbols only when explicitly requested."""

from importlib import import_module


def optional_symbol(module: str, name: str):
    try:
        return getattr(import_module(module), name)
    except ModuleNotFoundError as exc:
        if exc.name not in {"custom_gaussian_rasterization", "hybrid_rasterization"}:
            raise
        return None
