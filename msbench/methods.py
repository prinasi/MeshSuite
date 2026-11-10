"""Built-in method metadata, usable without PyTorch or CUDA imports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    vendor_subdir: str
    package: str
    adapter_module: str
    cuda_backends: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    training_module: str | None = None
    # Third-party modules the method imports at runtime; MeshSuite does not build them.
    runtime_modules: tuple[str, ...] = ()


METHODS = {
    "exmesh": MethodSpec(
        "exmesh", "exmesh", "msbench.vendor.exmesh",
        "msbench.renderers.exmesh_adapter", (), (),
        "msbench.trainers.exmesh_method",
        runtime_modules=("nvdiffrast.torch", "xatlas"),
    ),
    "2dts": MethodSpec(
        "2dts", "d2ts", "msbench.vendor.d2ts.diff_recon",
        "msbench.renderers.d2ts_adapter", ("2dts", "simple-knn"), ("d2ts",),
    ),
    "triangle-splatting": MethodSpec(
        "triangle-splatting", "triangle_splatting", "msbench.vendor.triangle_splatting",
        "msbench.renderers.triangle_splatting_adapter",
        ("triangle-splatting", "simple-knn"), ("trianglesplatting",),
        "msbench.trainers.triangle_splatting_method",
    ),
    "mesh-splatting": MethodSpec(
        "mesh-splatting", "mesh_splatting", "msbench.vendor.mesh_splatting",
        "msbench.renderers.mesh_splatting_adapter",
        ("mesh-splatting", "simple-knn"), ("meshsplatting",),
        "msbench.trainers.mesh_splatting_method",
    ),
    "t2splat": MethodSpec(
        "t2splat", "t2splat", "msbench.vendor.t2splat",
        "msbench.renderers.t2splat_adapter",
        ("t2splat", "simple-knn"), ("t2-splat", "t^2-splat", "t²-splat"),
    ),
    "diffsoup": MethodSpec(
        "diffsoup", "diffsoup", "msbench.vendor.diffsoup",
        "msbench.renderers.diffsoup_adapter", ("diffsoup",), (),
        "msbench.trainers.diffsoup_native",
    ),
}

CUDA_MODULES = {
    "2dts": "msbench.vendor._cmod.d2ts_rasterization._C",
    "triangle-splatting": "msbench.vendor._cmod.triangle_splatting_rasterization._C",
    "mesh-splatting": "msbench.vendor._cmod.mesh_splatting_rasterization._C",
    "t2splat": "msbench.vendor._cmod.t2splat_rasterization._C",
    "simple-knn": "msbench.vendor._cmod.simple_knn._C",
    "diffsoup": "msbench.vendor.diffsoup.diffsoup._core",
}


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


_ALIASES = {
    normalize_name(alias): name
    for name, spec in METHODS.items()
    for alias in (name, *spec.aliases)
}


def canonical_method_name(name: str) -> str:
    key = normalize_name(name)
    if key not in _ALIASES:
        raise KeyError(f"Unknown backend '{name}'. Available backends: {', '.join(sorted(METHODS))}")
    return _ALIASES[key]


def resolve_cuda_backends(selection: str | None) -> frozenset[str]:
    """Validate a build selection and include each method's CUDA dependencies."""
    if selection is None or normalize_name(selection) == "all":
        return frozenset(CUDA_MODULES)
    requested = [normalize_name(item) for item in selection.split(",")]
    if not all(requested):
        raise ValueError("MSBENCH_CUDA_BACKENDS must contain nonempty backend names.")
    result = set()
    for item in requested:
        name = _ALIASES.get(item, item)
        if name not in CUDA_MODULES:
            raise ValueError(
                f"Unknown CUDA backend '{item}'. Available: {', '.join(sorted(CUDA_MODULES))}, all"
            )
        result.update(METHODS[name].cuda_backends if name in METHODS else (name,))
    return frozenset(result)
