"""Vendored simple-knn CUDA wrapper."""

from __future__ import annotations

import torch

try:
    from . import _C
    _C_IMPORT_ERROR = None
except ImportError as exc:
    try:
        import simple_knn._C as _C
        _C_IMPORT_ERROR = None
    except ImportError:
        _C = None
        _C_IMPORT_ERROR = exc


def _require_cuda() -> None:
    if _C is None:
        raise RuntimeError(
            "simple_knn CUDA extension is not compiled or cannot load. "
            "Build the selected method with MSBENCH_CUDA_BACKENDS=<method>; "
            "implicit torch.cdist fallback is disabled for benchmark integrity."
        ) from _C_IMPORT_ERROR


def distCUDA2(points: torch.Tensor) -> torch.Tensor:
    """Return the mean squared distance to each point's 3 nearest neighbors."""
    _require_cuda()
    return _C.distCUDA2(points)


def nearestNeighbor(points: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
    """Return nearest-neighbor indices, optionally grouped by batch."""
    _require_cuda()
    return _C.nearestNeighbor(points, batch_size)
