"""Bake DiffSoup's neural textures for standard engine deployment.

DiffSoup stores a feature/opacity lattice per triangle and decodes it in the
shader with an MLP that also takes the view direction, so it has no appearance a
conventional material can show. This module fixes that appearance, like 2DGS's
TSDF colours: the decoder output of every lattice texel is averaged over the
training views that see it, weighted by the pixels it covers. The result is a
view-independent RGBA atlas on the unchanged triangle soup, drawn with a plain
textured alpha-cutout material. No held-out view is used.

The package holds only what the standard mesh needs (positions, indices, UVs and
the atlas), so its size and memory are those of the deployed mesh.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from msbench.unity_assets import (
    EXPORT_CONTRACT_REVISION,
    PACKAGE_SUFFIX,
    SCHEMA_VERSION,
    _PackageWriter,
    _sha256,
)

ATLAS_FILE = "atlas.png"
APPEARANCE = "baked_texture_cutout"
# One texel of padding on each side of the (2^R + 1)^2 lattice square.
TILE_PADDING = 1
ALPHA_THRESHOLD = 0.5
_SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199
_SH_C2 = (1.0925484305920792, 0.31539156525252005, 0.5462742152960396)
_SHADE_CHUNK = 4_000_000


def lattice_xy(level: int, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Return the lattice coordinates of DiffSoup's samples, in storage order.

    Sample ``q = n (n + 1) / 2 + y`` with ``n = x + y`` sits at barycentrics
    ``b0 = x / 2^level`` and ``b1 = y / 2^level``.
    """
    res = 1 << level
    y = torch.cat([torch.arange(n + 1, device=device) for n in range(res + 1)])
    n = torch.cat([torch.full((n + 1,), n, device=device) for n in range(res + 1)])
    return n - y, y


def atlas_layout(num_triangles: int, level: int) -> dict[str, int]:
    """Lay one square tile per triangle out on a near-square grid."""
    if num_triangles <= 0:
        raise ValueError("atlas needs at least one triangle")
    tile = (1 << level) + 1 + 2 * TILE_PADDING
    columns = int(math.ceil(math.sqrt(num_triangles)))
    rows = int(math.ceil(num_triangles / columns))
    return {
        "tile": tile,
        "columns": columns,
        "rows": rows,
        "width": columns * tile,
        "height": rows * tile,
    }


def atlas_uvs(num_triangles: int, level: int) -> torch.Tensor:
    """Return ``(T, 3, 2)`` UVs of each face's corners, in Unity's bottom-up v.

    Corner ``i`` has barycentric ``b_i = 1``, so the corners sit on the lattice at
    ``(res, 0)``, ``(0, res)`` and ``(0, 0)``. UVs address texel centres, which
    makes bilinear filtering interpolate between lattice samples.
    """
    layout = atlas_layout(num_triangles, level)
    res = 1 << level
    index = torch.arange(num_triangles)
    left = (index % layout["columns"]) * layout["tile"] + TILE_PADDING
    top = (index // layout["columns"]) * layout["tile"] + TILE_PADDING
    corners = torch.tensor([[res, 0], [0, res], [0, 0]], dtype=torch.float64)
    columns = left[:, None].double() + corners[:, 0]
    rows = top[:, None].double() + corners[:, 1]
    u = (columns + 0.5) / layout["width"]
    v = 1.0 - (rows + 0.5) / layout["height"]
    return torch.stack([u, v], dim=-1).float()


def build_atlas(rgba: torch.Tensor, level: int) -> np.ndarray:
    """Pack per-texel ``(T, S, 4)`` RGBA in [0, 1] into a top-down uint8 atlas.

    A square tile has one more row of samples than the triangular lattice needs.
    DiffSoup interpolates linearly over micro-triangles while hardware filtering
    is bilinear over square cells, so the row just past the hypotenuse is filled
    with the planar extrapolation ``a(x, y) = a(x-1, y) + a(x, y-1) - a(x-1, y-1)``.
    Bilinear filtering then reproduces the linear interpolation on those cells.
    The outer padding repeats the nearest edge sample.
    """
    num_triangles = rgba.shape[0]
    res = 1 << level
    x, y = lattice_xy(level)
    if rgba.shape[1] != x.numel() or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"expected (T, {x.numel()}, 4) RGBA, got {tuple(rgba.shape)}")
    layout = atlas_layout(num_triangles, level)
    tile = layout["tile"]

    grid = torch.zeros(num_triangles, res + 2, res + 2, 4, dtype=torch.float32)
    grid[:, x, y] = rgba.float().cpu()
    band_x = torch.arange(1, res + 1)
    band_y = res + 1 - band_x
    grid[:, band_x, band_y] = (
        grid[:, band_x - 1, band_y] + grid[:, band_x, band_y - 1] - grid[:, band_x - 1, band_y - 1]
    ).clamp(0.0, 1.0)
    grid[:, res + 1, 0] = grid[:, res, 0]
    grid[:, 0, res + 1] = grid[:, 0, res]

    tiles = torch.zeros(num_triangles, tile, tile, 4, dtype=torch.float32)
    tiles[:, TILE_PADDING:, TILE_PADDING:] = grid
    # Tiles are indexed [x, y]; replicate the x = 0 and y = 0 edges into the padding.
    tiles[:, 0, :] = tiles[:, 1, :]
    tiles[:, :, 0] = tiles[:, :, 1]
    tiles_u8 = (tiles * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()

    atlas = np.zeros((layout["height"], layout["width"], 4), dtype=np.uint8)
    for index in range(num_triangles):
        row, column = divmod(index, layout["columns"])
        top, left = row * tile, column * tile
        # Atlas rows follow lattice y and atlas columns follow lattice x.
        atlas[top : top + tile, left : left + tile] = tiles_u8[index].transpose(1, 0, 2)
    return atlas


def _eval_sh2(direction: torch.Tensor) -> torch.Tensor:
    """Order-2 SH basis, as diffsoup_core/cuda/encoding.cu ``eval_sh2``."""
    x, y, z = direction[..., 0], direction[..., 1], direction[..., 2]
    xx, yy, zz = x * x, y * y, z * z
    return torch.stack(
        [
            torch.full_like(x, _SH_C0),
            -_SH_C1 * y,
            _SH_C1 * z,
            -_SH_C1 * x,
            _SH_C2[0] * x * y,
            -_SH_C2[0] * y * z,
            _SH_C2[1] * (2.0 * zz - xx - yy),
            -_SH_C2[0] * x * z,
            _SH_C2[2] * (xx - yy),
        ],
        dim=-1,
    )


def _view_directions(ndc_xy: torch.Tensor, inv_mvp: torch.Tensor) -> torch.Tensor:
    """Surface-to-camera direction, as encoding.cu ``compute_view_direction``."""
    ones = torch.ones_like(ndc_xy[:, :1])
    near = torch.cat([ndc_xy, -ones, ones], dim=-1) @ inv_mvp.T
    far = torch.cat([ndc_xy, ones, ones], dim=-1) @ inv_mvp.T
    direction = near[:, :3] / near[:, 3:] - far[:, :3] / far[:, 3:]
    return torch.nn.functional.normalize(direction, dim=-1)


@torch.no_grad()
def bake_lattice_colors(
    checkpoint: dict[str, Any],
    mvps: list[torch.Tensor],
    *,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Average the decoded colour of each lattice texel over ``mvps``.

    Texels no view sees keep a usable colour: the decoder is evaluated with their
    triangle's mean view encoding, or the mean of all views if none sees the
    triangle.

    Returns ``(T, S, 3)`` colours in [0, 1] and coverage statistics.
    """
    import msbench.vendor.diffsoup as ds

    device = torch.device(device)
    vertices = checkpoint["V"].to(device=device, dtype=torch.float32).contiguous()
    faces = checkpoint["F"].to(device=device, dtype=torch.int32).contiguous()
    features = checkpoint["feat_acc"].to(device=device, dtype=torch.float32).contiguous()
    alpha = checkpoint["alpha_acc"].to(device=device, dtype=torch.float32).contiguous()
    level = int(checkpoint["Rmax"])
    height, width = int(checkpoint["H"]), int(checkpoint["W"])
    res = 1 << level
    num_triangles, samples = features.shape[:2]

    color_mlp = ds.ColorMLP(
        input_dim=features.shape[-1] + 9,
        hidden_dim=checkpoint["color_mlp"]["mlp.0.weight"].shape[0],
        n_layers=sum(key.endswith(".weight") for key in checkpoint["color_mlp"]) - 1,
        output_dim=3,
    ).to(device)
    color_mlp.load_state_dict(checkpoint["color_mlp"])
    color_mlp.eval()

    def decode(texel_features: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([texel_features, encoding], dim=-1)
        return color_mlp.forward(inputs.view(1, 1, -1, inputs.shape[-1]), mask=None).view(-1, 3).clamp(0.0, 1.0)

    x, y = lattice_xy(level, device=device)
    sample_of = torch.full((res + 1, res + 1), -1, device=device, dtype=torch.long)
    sample_of[x, y] = torch.arange(samples, device=device)
    b0, b1 = x.float() / res, y.float() / res
    corners = vertices[faces.long()]
    # Native barycentrics are perspective-correct, so they locate world points.
    positions = (
        b0[None, :, None] * corners[:, None, 0]
        + b1[None, :, None] * corners[:, None, 1]
        + (1.0 - b0 - b1)[None, :, None] * corners[:, None, 2]
    )

    color_sum = torch.zeros(num_triangles, samples, 3, device=device)
    weight = torch.zeros(num_triangles, samples, device=device)
    encoding_sum = torch.zeros(num_triangles, 9, device=device)
    triangle_weight = torch.zeros(num_triangles, device=device)
    homogeneous = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=-1)

    for mvp in mvps:
        mvp = mvp.to(device=device, dtype=torch.float32)
        inv_mvp = torch.inverse(mvp)
        clip = (homogeneous @ mvp.T).unsqueeze(0).contiguous()
        raster = ds.rasterize_multires_triangle_alpha(
            (height, width), clip, faces, level=level, alpha_src=alpha, stochastic=False
        )[0]
        visible = raster[..., 3] > 0
        triangle = (raster[..., 3][visible] - 1).long()
        lattice_x = (raster[..., 0][visible] * res).round().long().clamp(0, res)
        lattice_y = (raster[..., 1][visible] * res).round().long().clamp(0, res)
        lattice_x = torch.minimum(lattice_x, res - lattice_y)
        keys, counts = (triangle * samples + sample_of[lattice_x, lattice_y]).unique(return_counts=True)
        texel_triangle, texel_sample = keys // samples, keys % samples
        counts = counts.to(torch.float32)

        projected = torch.cat(
            [positions[texel_triangle, texel_sample], torch.ones_like(counts)[:, None]], dim=-1
        ) @ mvp.T
        encoding = _eval_sh2(_view_directions(projected[:, :2] / projected[:, 3:], inv_mvp))
        colors = decode(features[texel_triangle, texel_sample], encoding)

        color_sum.index_put_((texel_triangle, texel_sample), colors * counts[:, None], accumulate=True)
        weight.index_put_((texel_triangle, texel_sample), counts, accumulate=True)
        encoding_sum.index_add_(0, texel_triangle, encoding * counts[:, None])
        triangle_weight.index_add_(0, texel_triangle, counts)

    if not float(triangle_weight.sum()) > 0.0:
        raise ValueError("no view sees any triangle; check the cameras")

    baked = color_sum / weight.clamp(min=1.0)[..., None]
    mean_encoding = encoding_sum.sum(dim=0) / triangle_weight.sum()
    triangle_encoding = torch.where(
        (triangle_weight > 0)[:, None],
        encoding_sum / triangle_weight.clamp(min=1.0)[:, None],
        mean_encoding[None],
    )
    unseen_triangle, unseen_sample = torch.nonzero(weight <= 0, as_tuple=True)
    for start in range(0, unseen_triangle.numel(), _SHADE_CHUNK):
        chunk_triangle = unseen_triangle[start : start + _SHADE_CHUNK]
        chunk_sample = unseen_sample[start : start + _SHADE_CHUNK]
        baked[chunk_triangle, chunk_sample] = decode(
            features[chunk_triangle, chunk_sample], triangle_encoding[chunk_triangle]
        )

    opaque = alpha[..., 0] > ALPHA_THRESHOLD
    stats = {
        "views": len(mvps),
        "texels_seen_fraction": float((weight > 0).float().mean()),
        "opaque_texels_seen_fraction": float((weight[opaque] > 0).float().mean()) if opaque.any() else 0.0,
        "triangles_seen_fraction": float((triangle_weight > 0).float().mean()),
    }
    return baked.cpu(), stats


def write_baked_package(
    checkpoint: dict[str, Any],
    checkpoint_path: str | Path,
    baked_rgb: torch.Tensor,
    stats: dict[str, float],
    output: str | Path,
    *,
    overwrite: bool = False,
    background_color: str | None = None,
) -> Path:
    """Write the standard engine deployment package of a baked DiffSoup model."""
    if background_color is None:
        background_color = "white" if checkpoint.get("white_background") else "black"
    output_path = Path(output).expanduser()
    if output_path.suffix != PACKAGE_SUFFIX:
        output_path = output_path.with_name(output_path.name + PACKAGE_SUFFIX)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Unity asset already exists: {output_path}; use --force to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_path)

    faces = checkpoint["F"].long()
    level = int(checkpoint["Rmax"])
    num_triangles = faces.shape[0]
    alpha = checkpoint["alpha_acc"].float().cpu()
    layout = atlas_layout(num_triangles, level)

    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent))
    try:
        (staging / "buffers").mkdir()
        atlas = build_atlas(torch.cat([baked_rgb, alpha], dim=-1), level)
        Image.fromarray(atlas, mode="RGBA").save(staging / ATLAS_FILE)

        # One vertex per face corner, so each corner can carry its tile's UV.
        writer = _PackageWriter(staging)
        writer.tensor("positions", checkpoint["V"].float()[faces].reshape(-1, 3), "per-corner triangle positions")
        writer.tensor(
            "indices",
            torch.arange(num_triangles * 3, dtype=torch.int32).reshape(-1, 3),
            "triangle indices",
            dtype=torch.int32,
        )
        writer.tensor("uvs", atlas_uvs(num_triangles, level).reshape(-1, 2), "per-corner atlas UVs (v up)")

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "export_contract_revision": EXPORT_CONTRACT_REVISION,
            "asset_kind": "msbench-unity-native",
            "method": "diffsoup",
            "source": {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
            },
            "coordinate_system": {
                "name": "msbench_world",
                "up_axis": "+y",
                "handedness": "source-defined",
                "unit_scale": 1.0,
            },
            "rendering": {
                "color_space": "native_image_code_values",
                "requires_method_renderer": False,
                "generic_mesh_equivalent": True,
                "unity_profile_name": "method-aware",
                "cuda_equivalent": False,
                "background_color": background_color,
                "renderer": "standard-engine-mesh",
                "primitive_topology": "indexed-triangles",
                "primitive_count": int(num_triangles),
                "standard_engine": {
                    "supported": True,
                    "appearance": APPEARANCE,
                    "view_dependent_appearance": False,
                    "opaque_depth_tested": True,
                    "alpha_test": ALPHA_THRESHOLD,
                    "atlas": {
                        "file": ATLAS_FILE,
                        "width": layout["width"],
                        "height": layout["height"],
                        "format": "rgba8-straight-alpha",
                        "filter": "bilinear",
                        "mipmaps": False,
                        "lattice_level": level,
                    },
                    "bake": {"split": "train", **stats},
                },
                "unity_method_aware_requires_per_camera_depth_sort": False,
                "unity_method_aware_status": "baked-cutout-mesh",
            },
            "buffers": {name: spec.to_dict() for name, spec in writer.buffers.items()},
            "buffer_list": [{"name": name, **spec.to_dict()} for name, spec in writer.buffers.items()],
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if output_path.exists():
        shutil.rmtree(output_path)
    staging.rename(output_path)
    return output_path
