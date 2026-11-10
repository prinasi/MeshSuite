from __future__ import annotations

import importlib.util
import hashlib
import os
import sys
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
BUILD_HELPER = ROOT / "msbench" / "vendor" / "build.py"


def _should_build_cuda() -> bool:
    """Decide whether to compile CUDA extensions.

    Decision priority:
      1. ``MSBENCH_SKIP_CUDA=1``  → always skip (fastest daily dev).
      2. ``MSBENCH_BUILD_CUDA=1`` → always build (explicit opt-in).
      3. ``MSBENCH_CUDA_BACKENDS`` selects a method and its dependencies.
      4. Otherwise install Python code without compiling any extensions.
    """
    skip = os.environ.get("MSBENCH_SKIP_CUDA", "").lower()
    if skip in ("1", "true", "yes"):
        return False

    force = os.environ.get("MSBENCH_BUILD_CUDA", "").lower()
    if force in ("1", "true", "yes"):
        return True

    return bool(os.environ.get("MSBENCH_CUDA_BACKENDS", "").strip())


kwargs: dict = dict(
    packages=find_packages(include=["msbench", "msbench.*"]),
)

# setuptools may otherwise reuse build/lib across interpreters and selected
# backends, leaking old binaries into a wheel for another method environment.
build_identity = [sys.executable, str(_should_build_cuda()),
                  os.getenv("MSBENCH_CUDA_BACKENDS", "all"),
                  os.getenv("TORCH_CUDA_ARCH_LIST", ""), os.getenv("MSBENCH_CUDA_ARCHS", "89"),
                  os.getenv("CUDA_HOME", ""), os.getenv("CUDACXX", "")]

if _should_build_cuda():
    spec = importlib.util.spec_from_file_location("msbench_vendor_build", BUILD_HELPER)
    build = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(build)
    import torch

    build_identity.extend([torch.__version__, str(torch.version.cuda)])
    kwargs["ext_modules"] = build.get_ext_modules()
    kwargs["cmdclass"] = {"build_ext": build.ParallelBuildExt}

    backends = os.environ.get("MSBENCH_CUDA_BACKENDS", "all")
    ccache = "yes" if os.environ.get("MSBENCH_CCACHE", "").lower() in ("1", "true", "yes") else "no"
    print(
        f"[MeshSuite] Building CUDA extensions (backends={backends}, ccache={ccache}).\n"
        "  Tip: set MSBENCH_SKIP_CUDA=1 to skip CUDA compilation for faster\n"
        "       editable installs when you only changed Python code."
    )
else:
    print(
        "[MeshSuite] Skipping CUDA extension build.\n"
        "  Python-only toolkit installed. To build CUDA extensions:\n"
        "    MSBENCH_CUDA_BACKENDS=triangle-splatting pip install '.[runtime,cuda]' --no-build-isolation\n"
        "    # explicitly compile all backends:\n"
        "    MSBENCH_BUILD_CUDA=1 pip install '.[runtime,cuda]' --no-build-isolation\n"
        "\n"
        "  Speed tips:\n"
        "    MSBENCH_CCACHE=1           # cache NVCC output (needs ccache)\n"
        "    MSBENCH_CUDA_BACKENDS=triangle-splatting,simple-knn  # build subset\n"
        "    MAX_BUILD_JOBS=8                 # parallel compilation\n"
    )

build_tag = hashlib.sha256("\n".join(build_identity).encode()).hexdigest()[:16]
kwargs["options"] = {"build": {"build_base": str(ROOT / "build" / build_tag)}}
setup(**kwargs)
