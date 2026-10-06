from .triangle_renderer import TriangleRenderer

__all__ = ["TriangleRenderer", "GaussianRenderer", "HybridRenderer"]
_EXPERIMENTAL_SYMBOLS = {
    "GaussianRenderer": ".gaussian_renderer",
    "HybridRenderer": ".hybrid_renderer",
}


def __getattr__(name):
    if name not in _EXPERIMENTAL_SYMBOLS:
        raise AttributeError(name)
    from .._optional import optional_symbol

    value = optional_symbol(__name__ + _EXPERIMENTAL_SYMBOLS[name], name)
    globals()[name] = value
    return value
