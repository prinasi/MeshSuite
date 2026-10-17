"""MeshSuite state and camera boundary for the ExMesh reference runtime."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

MODEL_FIELDS = ("vertices", "faces", "uvs", "uv_indices", "vmapping", "texture", "texture_mask")
STAT_FIELDS = ("image_size", "degeneracy_ratio", "mask_grad_ema", "vertices_ref")


def validate_model_state(state):
    missing = set(MODEL_FIELDS) - state.keys()
    if missing:
        raise ValueError(f"ExMesh checkpoint missing fields: {sorted(missing)}")
    v, f, uv, uf, mapping, tex = [
        state[k] for k in ("vertices", "faces", "uvs", "uv_indices", "vmapping", "texture")
    ]
    if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
        raise ValueError("ExMesh vertices must be nonempty [V,3]")
    if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0 or f.is_floating_point():
        raise ValueError("ExMesh faces must be nonempty integer [F,3]")
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError("ExMesh face indices out of bounds")
    if uv.ndim != 2 or uv.shape[1] != 2 or uf.shape != f.shape or uf.is_floating_point():
        raise ValueError("ExMesh UV coordinates/face indices have invalid shapes or dtype")
    if uf.min() < 0 or uf.max() >= len(uv):
        raise ValueError("ExMesh UV face indices out of bounds")
    if (
        mapping.shape != (len(uv),)
        or mapping.is_floating_point()
        or mapping.min() < 0
        or mapping.max() >= len(v)
    ):
        raise ValueError("ExMesh UV-to-vertex mapping is invalid")
    if not torch.equal(mapping[uf.long()], f):
        raise ValueError("ExMesh geometry and UV topology disagree")
    if tex.ndim != 3 or tex.shape[0] != 3 or state["texture_mask"].shape != tex.shape[1:]:
        raise ValueError("ExMesh texture must be [3,H,W] with a matching mask")
    for name in ("vertices", "uvs", "texture", "texture_mask"):
        if not torch.isfinite(state[name]).all():
            raise ValueError(f"ExMesh {name} contains nonfinite values")


def model_state(model):
    state = {k: getattr(model, "_" + k).detach().cpu().clone() for k in MODEL_FIELDS}
    state["vertices_color"] = model._vertices_color.detach().cpu().clone()
    return state


def restore_model(state, *, device="cuda"):
    from .mesh_model import MeshModel

    validate_model_state(state)
    model = MeshModel()
    for name in MODEL_FIELDS:
        value = state[name].to(device).detach().clone()
        if name in {"vertices", "texture"}:
            value = torch.nn.Parameter(value)
        setattr(model, "_" + name, value)
    model._vertices_color = state.get("vertices_color", torch.zeros_like(state["vertices"])).to(
        device
    )
    model._number_of_faces = len(model._faces)
    model.image_size = torch.zeros(len(model._faces), device=device)
    model.degeneracy_ratio = torch.zeros(len(model._faces), device=device)
    model.mask_grad_ema = torch.zeros(len(model._vertices), device=device)
    model.vertices_ref = model._vertices.detach().clone()
    return model


def initialize_mesh(path, texture_resolution, background):
    import open3d as o3d
    from .geometry import BasicMesh, extract_uv_map
    from .mesh_model import MeshModel

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ExMesh requires an initial coarse mesh: {path}")
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_triangles():
        raise ValueError(f"Initial ExMesh mesh has no faces: {path}")
    vertices = np.asarray(mesh.vertices)
    colors = (
        np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.full_like(vertices, 0.5)
    )
    mesh.compute_vertex_normals()
    basic = BasicMesh(
        vertices,
        np.asarray(mesh.triangles),
        colors,
        np.asarray(mesh.vertex_normals),
        None,
        None,
        None,
    )
    basic = extract_uv_map(basic)
    model = MeshModel()
    model.create_from_mesh(basic, texture_resolution, background)
    model.vertices_ref = model.get_vertices.detach().clone()
    return model


def native_camera(cameras, index=0):
    """Retain ExMesh's centered projection and column W2C convention exactly."""
    k = cameras.Ks[index]
    near, far = cameras.near, cameras.far
    proj = torch.zeros((4, 4), device=k.device, dtype=k.dtype)
    proj[0, 0] = 2 * k[0, 0] / cameras.width
    proj[1, 1] = 2 * k[1, 1] / cameras.height
    proj[3, 2] = 1
    proj[2, 2] = far / (far - near)
    proj[2, 3] = -(far * near) / (far - near)
    return SimpleNamespace(
        world_view_transform=cameras.viewmats[index],
        projection_matrix=proj,
        image_height=cameras.height,
        image_width=cameras.width,
    )
