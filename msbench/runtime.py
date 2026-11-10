"""CUDA preflight checks, executed only inside a method's Python environment."""

from __future__ import annotations

import importlib
import inspect

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
        _require_rdel(name)
    if name == "t2splat":
        # T²-Splat always runs restricted Delaunay (by default at densify_until_iter + 1000)
        # and passes the face-selector keywords merged into submodules/effrdel.
        _require_rdel(name, keywords=("return_debug", "face_selector"))


def _require_rdel(name: str, keywords: tuple[str, ...] = ()) -> None:
    """Require effrdel's rdel.run, accepting ``keywords``, before training starts."""
    install = "pip install --no-build-isolation ./submodules/effrdel"
    try:
        rdel = importlib.import_module("rdel")
        if not callable(getattr(rdel, "run", None)):
            raise ImportError("rdel module does not provide run")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"Method '{name}' runs restricted Delaunay but rdel (effrdel) cannot load in this "
            f"environment. Install it first: {install}. Original error: {exc}"
        ) from exc
    missing = [key for key in keywords if key not in inspect.signature(rdel.run).parameters]
    if missing:
        raise RuntimeError(
            f"Method '{name}' calls rdel.run with {', '.join(missing)}, which the installed "
            f"effrdel ({getattr(rdel, '__file__', 'rdel')}) predates. Reinstall it from this "
            f"checkout: {install}"
        )
