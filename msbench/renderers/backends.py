"""Bundled renderer backend registry.

MeshSplatBench vendors the minimal runtime and CUDA sources needed by each
supported renderer under ``msbench.vendor``.  The registry mirrors the
Splatwizard style: methods are selected by name while concrete Python/CUDA
modules are loaded lazily from a package-local namespace.
"""

from __future__ import annotations

import importlib.util
from importlib import import_module
from dataclasses import dataclass
from pathlib import Path

from msbench.methods import CUDA_MODULES, METHODS, canonical_method_name


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_VENDOR_ROOT = _PACKAGE_ROOT / "vendor"


class BackendUnavailableError(ImportError):
    """Raised when a bundled renderer backend cannot be imported."""


@dataclass(frozen=True)
class ModuleStatus:
    """Import status for a Python or CUDA extension module."""

    name: str
    available: bool
    origin: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class BackendSpec:
    """Configuration for one bundled renderer backend."""

    name: str
    package: str
    extension_modules: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    vendor_subdir: str | None = None

    @property
    def vendor_path(self) -> Path:
        """Filesystem location of the vendored backend package."""
        return _VENDOR_ROOT / (self.vendor_subdir or self.name.replace("-", "_"))

    def repo_root(self, override: str | Path | None = None) -> Path:
        """Compatibility shim for older adapter constructors.

        ``override`` is still accepted but no longer required.  New code should
        use the bundled package paths reported by ``status()``.
        """
        if override is not None:
            return Path(override).expanduser().resolve()
        return self.vendor_path

    def status(self, override: str | Path | None = None) -> dict[str, object]:
        """Return a lightweight availability report for this bundled backend."""
        modules = [self.package, *self.extension_modules]
        module_status = [_probe_module(module_name) for module_name in modules]
        path = self.repo_root(override)
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "bundled": True,
            "repo_root": str(path),
            "repo_exists": path.exists(),
            "python_paths": [str(path)],
            "modules": [s.__dict__ for s in module_status],
        }


def _probe_module(module_name: str) -> ModuleStatus:
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return ModuleStatus(name=module_name, available=False)
        if module_name.endswith(("._C", "._core")):
            # A Python wrapper can exist even when the binary is missing or
            # was compiled against a different PyTorch/CUDA ABI.
            import_module(module_name)
        return ModuleStatus(
            name=module_name,
            available=True,
            origin=getattr(spec, "origin", None),
        )
    except Exception as exc:  # pragma: no cover - defensive diagnostics
        return ModuleStatus(name=module_name, available=False, error=str(exc))


BACKENDS: dict[str, BackendSpec] = {
    name: BackendSpec(
        name=name,
        package=spec.package,
        extension_modules=tuple(CUDA_MODULES[tag] for tag in spec.cuda_backends),
        aliases=spec.aliases,
        vendor_subdir=spec.vendor_subdir,
    )
    for name, spec in METHODS.items()
}


def canonical_backend_name(name: str) -> str:
    """Return the canonical backend name for a user-provided name or alias."""
    return canonical_method_name(name)


def get_backend(name: str) -> BackendSpec:
    """Return a backend specification by canonical name or alias."""
    return BACKENDS[canonical_backend_name(name)]


def list_backends() -> list[str]:
    """Return all canonical backend names."""
    return sorted(BACKENDS)
