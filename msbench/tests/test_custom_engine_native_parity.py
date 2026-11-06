"""The Custom shaders must keep the native renderers' rules: deleting one costs dB."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from msbench.unity_assets import export_triasset
from tools.validate_unity_triasset_cpu import validate_triasset

SHADERS = Path(__file__).resolve().parents[2] / "unity/Assets/MeshSuite/Shaders"


def _compact(name: str) -> str:
    return "".join((SHADERS / name).read_text().split())


def _d2ts_checkpoint(path: Path, gamma: float) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    state = {
        "_vertex": torch.zeros(1, 3, 3),
        "_opacity": torch.zeros(1, 1),
        "_f_dc": torch.zeros(1, 3, 1, 3),
        "_f_rest": torch.zeros(1, 3, 15, 3),
    }
    torch.save((state, None, None, gamma), path / "30000.ckpt")
    return path


@pytest.mark.parametrize(
    "rule",
    [
        "EvalSh3(tri,normalize(center.zzz-_WorldSpaceCameraPos))",
        "o.edgeScale=perimeter/max(float3(a,b,c),1e-6);",
        "floatphi=saturate(min(scaled.x,min(scaled.y,scaled.z)));",
        "clipCenter.w<=0.2||o.opacity<0.01||inradius<1.0||extent<1.0||extent>1600.0||cosine<0.001",
    ],
)
def test_triangle_splatting_shader_keeps_the_native_rules(rule: str):
    assert rule in _compact("TriangleSplattingMetalBaseline.shader")


@pytest.mark.parametrize(
    "rule",
    [
        "pow(-2.0*log(1.0/255.0),0.5/max(_Gamma,1e-4))",
        "if(_Mode==2){float3pc=(p0+p1+p2)/3.0;p=pc+(p-pc)*Dilation();}",
        "floatecc=(1-3*min(i.bp.x,min(i.bp.y,i.bp.z)))*Dilation();",
        "clip(cov-1.0/255.0);",
    ],
)
def test_2dts_shader_keeps_the_native_rules(rule: str):
    assert rule in _compact("MethodSpecificSplat.shader")


def test_mesh_splatting_mode_keeps_its_own_coverage():
    shader = _compact("MethodSpecificSplat.shader")
    assert "cov=pow(max(3*min(i.b.x,min(i.b.y,i.b.z)),1e-6),s);" in shader
    assert "if(_Mode!=2)clip(a-1.0/255.0);" in shader


@pytest.mark.parametrize("gamma", [1.0, 1.7])
def test_tuple_checkpoint_gets_the_native_rescale_at_any_gamma(tmp_path: Path, gamma: float):
    package = export_triasset("2dts", _d2ts_checkpoint(tmp_path / "ckpt", gamma), tmp_path / "scene")
    rendering = json.loads(package.manifest_path.read_text())["rendering"]
    assert rendering["gamma_rescale"] is True
    assert rendering["gamma_rescale_source"] == "native_default"
    assert rendering["gamma_vertex_rescale"] < 1.0


def test_validator_marks_assets_with_the_inferred_rescale_as_stale(tmp_path: Path):
    package = export_triasset("2dts", _d2ts_checkpoint(tmp_path / "ckpt", 1.0), tmp_path / "scene")
    validate_triasset(package.manifest_path.parent)
    manifest = json.loads(package.manifest_path.read_text())
    manifest["rendering"].update(
        gamma_rescale=False, gamma_rescale_source="legacy_inferred_from_gamma", gamma_vertex_rescale=1.0
    )
    package.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inferred from gamma"):
        validate_triasset(package.manifest_path.parent)
