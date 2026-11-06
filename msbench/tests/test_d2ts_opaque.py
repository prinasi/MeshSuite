"""Mesh-oriented 2DTS: the opacity threshold at export and the Unity wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from msbench.unity_assets import D2TS_SOLID_OPACITY_LOGIT, export_triasset
from tools.validate_unity_triasset_cpu import validate_triasset

REPO = Path(__file__).resolve().parents[2]


def _write_checkpoint(path: Path, opacity: list[float], *, gamma: float = 50.0) -> None:
    count = len(opacity)
    state = {
        "_vertex": torch.arange(count * 9, dtype=torch.float32).reshape(count, 3, 3),
        "_opacity": torch.logit(torch.tensor(opacity)).reshape(count, 1),
        "_f_dc": torch.arange(count * 3, dtype=torch.float32).reshape(count, 1, 3),
        "_f_rest": torch.zeros(count, 15, 3),
    }
    torch.save((state, {}, None, gamma), path)


def test_export_keeps_only_the_triangles_the_native_renderer_draws(tmp_path: Path):
    checkpoint = tmp_path / "40000.ckpt"
    _write_checkpoint(checkpoint, [0.9, 0.2, 0.3, 0.29, 0.5])

    package = export_triasset("2dts", checkpoint, tmp_path / "solid", d2ts_ste_threshold=0.3)

    manifest = json.loads(package.manifest_path.read_text())
    rendering = manifest["rendering"]
    assert rendering["primitive_count"] == 3
    assert rendering["source_primitive_count"] == 5
    assert rendering["ste_threshold"] == 0.3
    assert rendering["terminal_solid_eligible"] is True
    assert rendering["unity_method_aware_requires_per_camera_depth_sort"] is False
    assert rendering["unity_method_aware_status"] == "opaque-terminal-solid"
    assert rendering["standard_engine"]["supported"] is True

    buffers = package.path / "buffers"
    positions = torch.from_numpy(_read(buffers / "positions.bin", "f4")).reshape(-1, 3, 3)
    colors = torch.from_numpy(_read(buffers / "sh_dc.bin", "f4")).reshape(-1, 3)
    logits = torch.from_numpy(_read(buffers / "opacity_logits.bin", "f4"))
    # Triangles 0, 2 and 4 survive; 0.3 itself is kept, as the native >= does.
    assert positions[:, 0, 0].tolist() == [0.0, 18.0, 36.0]
    assert colors[:, 0].tolist() == [0.0, 6.0, 12.0]
    assert logits.tolist() == [D2TS_SOLID_OPACITY_LOGIT] * 3
    assert _read(buffers / "indices.bin", "i4").tolist() == list(range(9))
    assert validate_triasset(package.path)["d2ts_ste_threshold"] == 0.3


def test_export_without_a_threshold_keeps_the_volumetric_contract(tmp_path: Path):
    checkpoint = tmp_path / "30000.ckpt"
    _write_checkpoint(checkpoint, [0.9, 0.2], gamma=1.0)

    package = export_triasset("2dts", checkpoint, tmp_path / "volumetric")

    rendering = json.loads(package.manifest_path.read_text())["rendering"]
    assert rendering["primitive_count"] == 2
    assert "terminal_solid_eligible" not in rendering
    assert rendering["unity_method_aware_requires_per_camera_depth_sort"] is True


@pytest.mark.parametrize(
    "opacity,gamma,threshold,message",
    [
        ([0.9], 1.0, 0.3, "not finished solidifying"),
        ([0.1, 0.2], 50.0, 0.3, "no 2dts triangle"),
        ([0.9], 50.0, 1.5, "must be in"),
    ],
)
def test_export_rejects_checkpoints_that_are_not_solid(tmp_path: Path, opacity, gamma, threshold, message):
    checkpoint = tmp_path / "x.ckpt"
    _write_checkpoint(checkpoint, opacity, gamma=gamma)
    with pytest.raises(ValueError, match=message):
        export_triasset("2dts", checkpoint, tmp_path / "out", d2ts_ste_threshold=threshold)


def test_unity_solid_pass_is_wired_and_included_in_players():
    renderer = (REPO / "unity/Assets/MeshSuite/Scripts/MethodSpecificSplatRenderer.cs").read_text()
    shader = (REPO / "unity/Assets/MeshSuite/Shaders/D2TSTerminalSolid.shader").read_text()
    meta = (REPO / "unity/Assets/MeshSuite/Shaders/D2TSTerminalSolid.shader.meta").read_text()
    graphics = (REPO / "unity/ProjectSettings/GraphicsSettings.asset").read_text()

    assert 'd2tsSolid ? "MeshSuite/D2TSTerminalSolid"' in renderer
    assert "terminal_solid_eligible" in renderer
    assert '"_SolidVertexScale"' in renderer and "_SolidVertexScale" in shader
    assert "ZWrite On" in shader and "Blend Off" in shader
    guid = next(line.split()[1] for line in meta.splitlines() if line.startswith("guid:"))
    assert f"guid: {guid}, type: 3" in graphics


def _read(path: Path, dtype: str):
    import numpy as np

    return np.fromfile(path, dtype=dtype)
