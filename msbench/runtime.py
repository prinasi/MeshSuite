"""CUDA preflight checks, executed only inside a method's Python environment."""

from __future__ import annotations

import importlib

from msbench.methods import CUDA_MODULES, METHODS, canonical_method_name


def optional_operator_status() -> dict[str, object]:
    """Record MeshSplatting's existing optional fused-SSIM selection."""
    try:
        module = importlib.import_module("fused_ssim")
        if not callable(getattr(module, "fused_ssim", None)):
            raise ImportError("fused_ssim module does not provide fused_ssim")
        return {"fused_ssim": {"available": True}}
    except (ImportError, OSError) as exc:
        return {"fused_ssim": {"available": False, "error": f"{type(exc).__name__}: {exc}"}}


def require_method_cuda(method: str, *, trainer_cfg: dict | None = None) -> None:
    """Require a working GPU and all compiled operators before training."""
    name = canonical_method_name(method)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"Method '{name}' requires an available CUDA device before training.")
    for tag in METHODS[name].cuda_backends:
        module = CUDA_MODULES[tag]
        try:
            importlib.import_module(module)
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                f"Method '{name}' requires CUDA operator '{module}', which could not load. "
                f"Build it in this environment with MSBENCH_CUDA_BACKENDS={name} "
                "pip install '.[runtime,cuda]' --no-build-isolation. "
                f"Original error: {exc}"
            ) from exc
    if name == "mesh-splatting" and int((trainer_cfg or {}).get("run_restricted_delaunay", -1)) >= 0:
        try:
            rdel = importlib.import_module("rdel")
            if not callable(getattr(rdel, "run", None)):
                raise ImportError("rdel module does not provide run")
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "MeshSplatting config enables restricted Delaunay but rdel (effrdel) "
                f"cannot load in this environment. Install submodules/effrdel first. Original error: {exc}"
            ) from exc
