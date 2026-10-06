"""Build selection and preflight invariants; no CUDA compilation required."""

import pytest
import torch

from msbench.methods import CUDA_MODULES, METHODS, resolve_cuda_backends
from msbench.runtime import require_method_cuda


@pytest.mark.parametrize("name", ["2dts", "mesh-splatting", "triangle-splatting"])
def test_selected_method_includes_knn_without_other_rasterizers(name):
    assert resolve_cuda_backends(name) == {name, "simple-knn"}


def test_cuda_aliases_all_and_invalid_names():
    assert resolve_cuda_backends("D2TS,mesh_splatting") == {"2dts", "mesh-splatting", "simple-knn"}
    assert resolve_cuda_backends("diffsoup") == {"diffsoup"}
    assert resolve_cuda_backends("all") == set(CUDA_MODULES)
    assert resolve_cuda_backends(None) == set(CUDA_MODULES)
    for value in ("trianlge-splatting", "diffsoup,typo", "", "2dts,"):
        with pytest.raises(ValueError):
            resolve_cuda_backends(value)


def test_training_preflight_rejects_missing_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="available CUDA device"):
        require_method_cuda("triangle-splatting")


def test_training_preflight_checks_only_selected_extensions(monkeypatch):
    import msbench.runtime as runtime

    loaded = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: loaded.append(name))
    require_method_cuda("triangle_splatting")
    assert loaded == [CUDA_MODULES[tag] for tag in METHODS["triangle-splatting"].cuda_backends]


def test_training_preflight_reports_binary_abi_failure(monkeypatch):
    import msbench.runtime as runtime

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def broken(name):
        raise ImportError("undefined symbol: ABI mismatch")

    monkeypatch.setattr(runtime.importlib, "import_module", broken)
    with pytest.raises(RuntimeError, match="ABI mismatch"):
        require_method_cuda("diffsoup")


def test_knn_never_falls_back_without_extension(monkeypatch):
    from msbench.vendor._cmod import simple_knn

    monkeypatch.setattr(simple_knn, "_C", None)
    for operation in (simple_knn.distCUDA2, simple_knn.nearestNeighbor):
        with pytest.raises(RuntimeError, match="implicit torch.cdist fallback is disabled"):
            operation(torch.zeros(2, 3))


def test_delaunay_dependency_is_checked_only_when_enabled(monkeypatch):
    import msbench.runtime as runtime

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    loaded = []

    def import_module(name):
        loaded.append(name)
        if name == "rdel":
            raise ImportError("not installed")

    monkeypatch.setattr(runtime.importlib, "import_module", import_module)
    require_method_cuda("mesh-splatting", trainer_cfg={"run_restricted_delaunay": -1})
    assert "rdel" not in loaded
    with pytest.raises(RuntimeError, match="restricted Delaunay.*rdel"):
        require_method_cuda("mesh-splatting", trainer_cfg={"run_restricted_delaunay": 11000})
    from types import SimpleNamespace

    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: SimpleNamespace(run=lambda: None))
    require_method_cuda("mesh-splatting", trainer_cfg={"run_restricted_delaunay": 11000})


def test_broken_plugin_does_not_hide_healthy_plugin(monkeypatch):
    from types import SimpleNamespace
    from msbench.core import plugins

    loaded = []

    def broken():
        raise ImportError("missing optional CUDA")

    monkeypatch.setattr(plugins, "import_module", lambda name: loaded.append(name))
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: [
        SimpleNamespace(name="broken", load=broken),
        SimpleNamespace(name="healthy", load=lambda: loaded.append("healthy")),
    ])
    errors = plugins.discover_plugins(["built-in"], "test.plugins")
    assert loaded == ["built-in", "healthy"]
    assert errors == {"test.plugins:broken": "ImportError: missing optional CUDA"}
