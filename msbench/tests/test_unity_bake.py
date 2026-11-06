"""CPU tests of DiffSoup's baked standard engine deployment package."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from msbench.unity_assets import export_triasset
from msbench.unity_bake import (
    APPEARANCE,
    ATLAS_FILE,
    atlas_layout,
    atlas_uvs,
    build_atlas,
    lattice_xy,
    write_baked_package,
)
from tools.validate_unity_triasset_cpu import validate_triasset

REPO = Path(__file__).resolve().parents[2]
LEVEL = 3
RES = 1 << LEVEL


def _sample_count(level: int) -> int:
    return ((1 << (level - 1)) + 1) * ((1 << level) + 1)


def _bilinear(atlas: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample a top-down atlas like a clamped, unfiltered-mip GPU texture."""
    height, width = atlas.shape[:2]
    column = uv[0] * width - 0.5
    row = (1.0 - uv[1]) * height - 0.5
    c0, r0 = int(np.floor(column)), int(np.floor(row))
    fc, fr = column - c0, row - r0
    texel = atlas.astype(np.float64)
    return (
        texel[r0, c0] * (1 - fc) * (1 - fr)
        + texel[r0, c0 + 1] * fc * (1 - fr)
        + texel[r0 + 1, c0] * (1 - fc) * fr
        + texel[r0 + 1, c0 + 1] * fc * fr
    )


def _lattice_linear(rgba: torch.Tensor, b0: float, b1: float) -> np.ndarray:
    """DiffSoup's lattice interpolation (diffsoup_core/cuda/multires.cu)."""
    rb0, rb1 = b0 * RES, b1 * RES
    x = min(int(np.floor(rb0)), RES - 1)
    y = min(int(np.floor(rb1)), RES - 1 - x)
    rb0, rb1 = rb0 - x, rb1 - y
    flip = rb0 + rb1 > 1
    points = [(x + 1, y), (x, y + 1), (x + flip, y + flip)]
    weights = [1 - rb1 if flip else rb0, 1 - rb0 if flip else rb1]
    weights.append(1 - sum(weights))
    index = [(a + b) * (a + b + 1) // 2 + b for a, b in points]
    return sum(w * rgba[i].double().numpy() for w, i in zip(weights, index))


def _checkpoint(tmp_path: Path, triangles: int = 5) -> tuple[dict, Path]:
    samples = _sample_count(LEVEL)
    checkpoint = {
        "V": torch.arange(triangles * 9, dtype=torch.float32).reshape(-1, 3),
        "F": torch.arange(triangles * 3, dtype=torch.int32).reshape(-1, 3),
        "feat_acc": torch.rand(triangles, samples, 7),
        "alpha_acc": torch.rand(triangles, samples, 1),
        "color_mlp": {"mlp.0.weight": torch.zeros(16, 16), "mlp.0.bias": torch.zeros(16)},
        "Rmin": 1,
        "Rmax": LEVEL,
        "feat_dim": 7,
    }
    path = tmp_path / "final_params.pt"
    torch.save(checkpoint, path)
    return checkpoint, path


def test_lattice_matches_diffsoup_storage_order():
    x, y = lattice_xy(LEVEL)
    assert x.numel() == _sample_count(LEVEL)
    assert ((x + y) <= RES).all()
    n = x + y
    assert torch.equal(n * (n + 1) // 2 + y, torch.arange(x.numel()))


def test_uvs_address_their_triangles_lattice_corners():
    torch.manual_seed(0)
    triangles = 7
    rgba = torch.rand(triangles, _sample_count(LEVEL), 4)
    atlas = build_atlas(rgba, LEVEL)
    uvs = atlas_uvs(triangles, LEVEL).double().numpy()
    x, y = lattice_xy(LEVEL)
    for triangle in range(triangles):
        for corner, (cx, cy) in enumerate([(RES, 0), (0, RES), (0, 0)]):
            sample = torch.nonzero((x == cx) & (y == cy)).item()
            expected = (rgba[triangle, sample] * 255).round().numpy()
            # float32 UVs land within a tiny fraction of the texel centre.
            np.testing.assert_allclose(_bilinear(atlas, uvs[triangle, corner]), expected, atol=1e-3)


def test_bilinear_atlas_sampling_tracks_lattice_interpolation():
    """On a smooth signal, hardware filtering must match the native interpolation."""
    x, y = lattice_xy(LEVEL)
    b0, b1 = x.float() / RES, y.float() / RES
    # Stay inside [0, 1] past the hypotenuse, where the extrapolated row is clamped.
    smooth = torch.stack([0.3 + 0.4 * b0, 0.3 + 0.4 * b1, 0.7 - 0.4 * b0, 0.25 + 0.5 * b0 * b1], dim=-1)
    rgba = torch.stack([smooth, smooth.flip(-1)], dim=0)
    atlas = build_atlas(rgba, LEVEL)
    uvs = atlas_uvs(2, LEVEL).double().numpy()

    rng = np.random.default_rng(0)
    for _ in range(300):
        triangle = int(rng.integers(2))
        bary = rng.dirichlet((1.0, 1.0, 1.0))
        uv = bary[0] * uvs[triangle, 0] + bary[1] * uvs[triangle, 1] + bary[2] * uvs[triangle, 2]
        expected = _lattice_linear(rgba[triangle], bary[0], bary[1]) * 255
        # Within 8-bit rounding plus the second-order term of a cell's two triangles.
        np.testing.assert_allclose(_bilinear(atlas, uv), expected, atol=1.5)


def test_atlas_cells_on_the_hypotenuse_stay_planar():
    x, y = lattice_xy(LEVEL)
    alpha = (0.1 + 0.05 * x.float() + 0.02 * y.float()) / 2
    rgba = torch.stack([alpha, alpha, alpha, alpha], dim=-1).unsqueeze(0)
    atlas = build_atlas(rgba, LEVEL).astype(np.float64)
    # Lattice (x, y) sits at atlas [y + 1, x + 1]; check the band past x + y = RES.
    for band_x in range(1, RES + 1):
        band_y = RES + 1 - band_x
        planar = (0.1 + 0.05 * band_x + 0.02 * band_y) / 2 * 255
        assert atlas[band_y + 1, band_x + 1, 3] == pytest.approx(planar, abs=1.0)


def test_baked_package_passes_the_contract_and_keeps_only_the_mesh(tmp_path: Path):
    checkpoint, path = _checkpoint(tmp_path)
    triangles = checkpoint["F"].shape[0]
    baked = torch.rand(triangles, _sample_count(LEVEL), 3)
    stats = {"views": 3, "texels_seen_fraction": 0.5}
    package = write_baked_package(checkpoint, path, baked, stats, tmp_path / "baked")

    assert package.name == "baked.triasset"
    assert validate_triasset(package)["status"] == "ok"
    manifest = json.loads((package / "manifest.json").read_text())
    rendering = manifest["rendering"]
    standard = rendering["standard_engine"]
    assert standard["supported"] is True
    assert standard["appearance"] == APPEARANCE
    assert standard["view_dependent_appearance"] is False
    assert standard["bake"]["split"] == "train"
    assert rendering["requires_method_renderer"] is False
    assert rendering["renderer"] == "standard-engine-mesh"
    assert set(manifest["buffers"]) == {"positions", "indices", "uvs"}
    assert manifest["buffers"]["uvs"]["shape"] == [triangles * 3, 2]

    layout = atlas_layout(triangles, LEVEL)
    with Image.open(package / ATLAS_FILE) as atlas:
        assert atlas.mode == "RGBA"
        assert atlas.size == (layout["width"], layout["height"])
        # Straight alpha: colours under transparent texels are kept for filtering.
        stored = np.asarray(atlas)
    assert np.array_equal(stored, build_atlas(torch.cat([baked, checkpoint["alpha_acc"]], dim=-1), LEVEL))

    positions = np.fromfile(package / "buffers" / "positions.bin", dtype=np.float32).reshape(-1, 3)
    np.testing.assert_array_equal(positions, checkpoint["V"][checkpoint["F"].long()].reshape(-1, 3).numpy())


def test_existing_package_needs_force(tmp_path: Path):
    checkpoint, path = _checkpoint(tmp_path, triangles=1)
    baked = torch.rand(1, _sample_count(LEVEL), 3)
    write_baked_package(checkpoint, path, baked, {}, tmp_path / "baked.triasset")
    with pytest.raises(FileExistsError):
        write_baked_package(checkpoint, path, baked, {}, tmp_path / "baked.triasset")
    write_baked_package(checkpoint, path, baked, {}, tmp_path / "baked.triasset", overwrite=True)


def test_validator_rejects_baked_package_without_its_atlas(tmp_path: Path):
    checkpoint, path = _checkpoint(tmp_path, triangles=1)
    package = write_baked_package(
        checkpoint, path, torch.rand(1, _sample_count(LEVEL), 3), {}, tmp_path / "baked"
    )
    (package / ATLAS_FILE).unlink()
    with pytest.raises(ValueError, match="atlas"):
        validate_triasset(package)


def test_raw_diffsoup_package_still_rejects_standard_engine(tmp_path: Path):
    checkpoint, path = _checkpoint(tmp_path, triangles=1)
    package = export_triasset("diffsoup", path, tmp_path / "raw")
    standard = json.loads(package.manifest_path.read_text())["rendering"]["standard_engine"]
    assert standard["supported"] is False
    assert "bake_diffsoup_standard_asset" in standard["reason"]


def test_cutout_shader_is_always_included_in_players():
    meta = (REPO / "unity/Assets/MeshSuite/Shaders/StandardTextureCutoutRaw.shader.meta").read_text()
    guid = next(line.split()[1] for line in meta.splitlines() if line.startswith("guid:"))
    graphics = (REPO / "unity/ProjectSettings/GraphicsSettings.asset").read_text()
    assert f"guid: {guid}, type: 3" in graphics


@pytest.mark.parametrize(
    "extra, baked, subdir",
    [
        (("--standard-engine",), True, "unity_baked"),
        (("--standard-engine", "--asset-subdir", "mine"), True, "mine"),
        ((), False, "unity_native"),
    ],
)
def test_eval_wrapper_bakes_only_diffsoup_standard_engine(tmp_path: Path, extra, baked, subdir):
    out = subprocess.run(
        [
            "bash", str(REPO / "single_unity_eval.sh"), "diffsoup", "mipnerf360/room", "0",
            "--unity", str(tmp_path / "unity"), "--unity-project", str(tmp_path),
            "--skip-player-build", "--unity-player", str(tmp_path / "player"),
            "--skip-unity-patch", "--dry-run", *extra,
        ],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    assert ("tools/bake_diffsoup_standard_asset.py" in out) is baked
    assert ("single_export_unity.sh" in out) is not baked
    assert f"--asset-subdir {subdir}" in out
    if baked:
        assert f"{subdir}/diffsoup.triasset" in out
        assert "--downscale 4" in out
