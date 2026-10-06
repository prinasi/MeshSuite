"""Build helpers for MeshSplatBench's bundled CUDA renderer modules."""

from __future__ import annotations

import copy
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension
from msbench.methods import resolve_cuda_backends
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
CSRC_ROOT = PACKAGE_ROOT / "vendor" / "_csrc"
BUILD_JOBS = os.getenv("MAX_BUILD_JOBS")

# Enable ccache for NVCC.  Requires ``ccache`` on PATH.
_USE_CCACHE = os.getenv("MSBENCH_CCACHE", "").lower() in ("1", "true", "yes")


class ParallelBuildExt(BuildExtension):
    """Build CUDA extensions in isolated temporary directories."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.parallel = True if BUILD_JOBS is None else int(BUILD_JOBS)

    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            self.build_cmake_extension(ext)
            return

        self_copy = copy.copy(self)
        self_copy.build_temp = os.path.join(self.build_temp, ext.name)
        BuildExtension.build_extension(self_copy, ext)

    def build_cmake_extension(self, ext: "CMakeExtension") -> None:
        cmake = shutil.which("cmake")
        if cmake is None:
            raise RuntimeError(
                "Building MeshSplatBench's bundled DiffSoup extension requires "
                "CMake. Install build tools with `pip install .[cuda]` before "
                "building with --no-build-isolation."
            )
        try:
            nanobind_dir = subprocess.check_output(
                [sys.executable, "-m", "nanobind", "--cmake_dir"],
                text=True,
            ).strip()
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Building MeshSplatBench's bundled DiffSoup extension requires "
                "nanobind in the Python environment used for the build. Install "
                "it with `pip install nanobind` before building with "
                "`--no-build-isolation`."
            ) from exc

        ext_fullpath = Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir = ext_fullpath.parent
        cfg = "Debug" if self.debug else "Release"
        build_temp = Path(self.build_temp, ext.name).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir.mkdir(parents=True, exist_ok=True)

        cuda_archs = os.getenv("MSBENCH_CUDA_ARCHS", "89")
        configure_args = [
            cmake,
            "-S",
            str(ext.sourcedir),
            "-B",
            str(build_temp),
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}",
            f"-Dnanobind_DIR={nanobind_dir}",
        ]

        nvcc_bin = os.getenv("CUDACXX")
        if not nvcc_bin and CUDA_HOME:
            candidate = Path(CUDA_HOME) / "bin" / "nvcc"
            if candidate.exists():
                nvcc_bin = str(candidate)
        if nvcc_bin:
            configure_args.append(f"-DCMAKE_CUDA_COMPILER={nvcc_bin}")
        if _USE_CCACHE and shutil.which("ccache"):
            configure_args.append(f"-DCMAKE_CUDA_COMPILER_LAUNCHER={shutil.which('ccache')}")
        if CUDA_HOME and Path(CUDA_HOME).exists():
            configure_args.append(f"-DCUDAToolkit_ROOT={CUDA_HOME}")
        build_args = [cmake, "--build", str(build_temp), "--config", cfg]
        if BUILD_JOBS is not None:
            build_args.extend(["--parallel", str(BUILD_JOBS)])

        subprocess.check_call(configure_args)
        subprocess.check_call(build_args)


class CMakeExtension(Extension):
    """A setuptools extension built by CMake instead of distutils."""

    def __init__(self, name: str, sourcedir: str | Path):
        super().__init__(name, sources=[])
        self.sourcedir = Path(sourcedir).resolve()


def _rel_sources(root: Path, files: list[str]) -> list[str]:
    return [
        (root / file).resolve().relative_to(PROJECT_ROOT).as_posix()
        for file in files
    ]


def get_ext_modules() -> list[Extension]:
    """Return all bundled CUDA/CMake extension modules.

    Environment variables that affect the build:

    ``MSBENCH_CUDA_BACKENDS``
        Comma-separated list of backend names to build.  If set, only the
        listed backends are compiled (all others are skipped).  Names are
        matched case-insensitively with underscores and hyphens treated as
        equivalent. Shared KNN dependencies are included automatically. Example::

            MSBENCH_CUDA_BACKENDS=triangle-splatting,simple-knn

    ``MSBENCH_CCACHE``
        Set to ``1`` to route NVCC through *ccache*.  Requires *ccache* on
        ``PATH``. Uses PyTorch's ``PYTORCH_NVCC`` wrapper and CMake's CUDA
        compiler launcher, without replacing the host compiler.

    ``MSBENCH_CUDA_ARCHS``
        Semicolon-separated CUDA architectures (default ``89``).
    """
    selected = resolve_cuda_backends(os.getenv("MSBENCH_CUDA_BACKENDS"))

    # Activate ccache for NVCC if requested (must happen before any build).
    if _USE_CCACHE and shutil.which("ccache"):
        nvcc = str(Path(CUDA_HOME) / "bin" / "nvcc") if CUDA_HOME else shutil.which("nvcc")
        if nvcc:
            os.environ.setdefault("PYTORCH_NVCC", f"{shlex.quote(shutil.which('ccache'))} {shlex.quote(nvcc)}")

    tri_root = CSRC_ROOT / "triangle_splatting_rasterization"
    mesh_root = CSRC_ROOT / "mesh_splatting_rasterization"
    d2ts_root = CSRC_ROOT / "d2ts_rasterization"
    knn_root = CSRC_ROOT / "simple_knn"
    diffsoup_root = CSRC_ROOT / "diffsoup_core"

    glm_include = str((tri_root / "third_party" / "glm").resolve())
    mesh_glm_include = str((mesh_root / "third_party" / "glm").resolve())

    extensions: list[Extension] = []

    # -- triangle-splatting rasterization -----------------------------
    if "triangle-splatting" in selected:
        extensions.append(CUDAExtension(
            name="msbench.vendor._cmod.triangle_splatting_rasterization._C",
            sources=_rel_sources(
                tri_root,
                [
                    "cuda_rasterizer/rasterizer_impl.cu",
                    "cuda_rasterizer/forward.cu",
                    "cuda_rasterizer/backward.cu",
                    "cuda_rasterizer/utils.cu",
                    "rasterize_points.cu",
                    "ext.cpp",
                ],
            ),
            extra_compile_args={"nvcc": [f"-I{glm_include}", "--use_fast_math"]},
        ))

    # -- mesh-splatting rasterization ---------------------------------
    if "mesh-splatting" in selected:
        extensions.append(CUDAExtension(
            name="msbench.vendor._cmod.mesh_splatting_rasterization._C",
            sources=_rel_sources(
                mesh_root,
                [
                    "cuda_rasterizer/rasterizer_impl.cu",
                    "cuda_rasterizer/forward.cu",
                    "cuda_rasterizer/backward.cu",
                    "cuda_rasterizer/utils.cu",
                    "cuda_rasterizer/adam.cu",
                    "rasterize_points.cu",
                    "ext.cpp",
                ],
            ),
            extra_compile_args={"nvcc": [f"-I{mesh_glm_include}", "--use_fast_math"]},
        ))

    # -- 2DTS rasterization ------------------------------------------
    if "2dts" in selected:
        extensions.append(CUDAExtension(
            name="msbench.vendor._cmod.d2ts_rasterization._C",
            sources=_rel_sources(
                d2ts_root,
                [
                    "src/rasterizer.cu",
                    "src/forward.cu",
                    "src/backward.cu",
                    "src/extension_interface.cu",
                    "ext.cpp",
                ],
            ),
            extra_compile_args={"nvcc": []},
        ))

    # -- simple-knn ---------------------------------------------------
    if "simple-knn" in selected:
        extensions.append(CUDAExtension(
            name="msbench.vendor._cmod.simple_knn._C",
            sources=_rel_sources(
                knn_root,
                [
                    "interface.cu",
                    "simple_knn.cu",
                    "ext.cpp",
                ],
            ),
            extra_compile_args={"nvcc": [], "cxx": []},
        ))

    # -- DiffSoup (CMake) ---------------------------------------------
    if "diffsoup" in selected:
        extensions.append(CMakeExtension(
            "msbench.vendor.diffsoup.diffsoup._core",
            diffsoup_root,
        ))

    return extensions
