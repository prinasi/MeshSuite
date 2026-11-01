"""Adapter for T²-Splat (adaptive-topology mesh splatting with texture residuals).

T²-Splat optimizes one connected triangle mesh (MeshSplatting rasterizer, Safe
RDT and Dynamic Topology Evolution). On NVS datasets it then refines per-face
residual Mip-textures on the frozen mesh. The adapter renders checkpoints
with the vendored T²-Splat CUDA rasterizer, including the textures, so image
metrics match upstream ``render.py``. Mesh export writes the trained mesh as is:
on DTU (mesh benchmark) it is cleaned as upstream ``create_ply.py --clean
--geometry-only`` does before Chamfer evaluation.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from msbench.core.cameras import CameraBatch
from msbench.core.registry import register
from msbench.core.stats import compute_triangle_stats
from msbench.renderers.backends import get_backend
from msbench.renderers.base import RendererAdapter, RenderOutput

# Bump when the exported mesh changes, so stale meshes are re-exported.
MESH_EXPORT_VERSION = 1


def load_t2splat_state(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a T²-Splat ``point_cloud_state_dict.pt`` (file or checkpoint dir) on the CPU."""
    checkpoint = Path(path).expanduser()
    state_path = checkpoint if checkpoint.is_file() else checkpoint / "point_cloud_state_dict.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {state_path}")
    state = torch.load(str(state_path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"T²-Splat checkpoint must contain a state dict: {state_path}")
    missing = [k for k in ("triangles_points", "_triangle_indices", "features_dc") if k not in state]
    if missing:
        raise ValueError(f"T²-Splat checkpoint is missing required tensors: {', '.join(missing)}")
    return state, state_path


def load_t2splat_primitive_checkpoint(path: str | Path):
    """Load exportable T²-Splat geometry (vertices, faces, SH-DC colours) on the CPU."""
    from msbench.primitives.mesh_triangle import IndexedMeshTriangle

    state, _ = load_t2splat_state(path)
    vertices = torch.as_tensor(state["triangles_points"], dtype=torch.float32).detach()
    faces = torch.as_tensor(state["_triangle_indices"], dtype=torch.long).detach()
    sh_dc = torch.as_tensor(state["features_dc"], dtype=torch.float32).detach()
    if vertices.dim() != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected T²-Splat vertices [V, 3], got {tuple(vertices.shape)}")
    if faces.dim() != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected T²-Splat faces [F, 3], got {tuple(faces.shape)}")
    if sh_dc.numel() != vertices.shape[0] * 3:
        raise ValueError(f"Expected one RGB SH-DC coefficient per vertex, got {tuple(sh_dc.shape)}")

    vertex_weights = None
    opacity = None
    if state.get("vertex_weight") is not None:
        # TriangleModel.load_parameters renders with opacity_floor = 0.999.
        logits = torch.as_tensor(state["vertex_weight"], dtype=torch.float32).reshape(-1)
        vertex_weights = 0.999 + (1.0 - 0.999) * torch.sigmoid(logits)
        opacity = vertex_weights[faces].amin(dim=1)
    sigma = torch.as_tensor(state.get("sigma", 0.0)).reshape(-1)
    return IndexedMeshTriangle(
        vertices=vertices,
        faces=faces,
        vertex_weights=vertex_weights,
        opacity=opacity,
        sigma=float(torch.exp(sigma[0]).item()) if sigma.numel() else 1.0,
        sh_coeffs=sh_dc.reshape(vertices.shape[0], 3),
    )


def clean_t2splat_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    min_faces_ratio: float = 0.01,
):
    """Upstream ``create_ply.clean_mesh``: drop components below ``min_faces_ratio``."""
    import trimesh

    from msbench.vendor.t2splat.create_ply import clean_mesh

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        vertex_colors=colors,
        process=False,
    )
    return clean_mesh(mesh, min_faces_ratio=min_faces_ratio, keep_largest=False)


@register("t2splat")
class T2SplatAdapter(RendererAdapter):
    """Renderer adapter for T²-Splat checkpoints (``ckpt/point_cloud/iteration_N``)."""

    def __init__(self, repo_root: str | Path | None = None):
        self._backend = get_backend("t2splat")
        self.repo_root = self._backend.repo_root(repo_root)
        self._initialized = False
        self._model = None
        self._checkpoint_path: str | None = None
        self._checkpoint_dir: Path | None = None
        self._model_path: Path | None = None
        self._loaded_iteration: int | None = None
        self._background_color = [0.0, 0.0, 0.0]
        self._background_color_override: list[float] | None = None
        # Upstream render.py and Stage II render with 4x supersampling.
        self._render_scaling = 4
        self._TriangleModel = None
        self._render = None
        self._getWorld2View2 = None
        self._getProjectionMatrix = None

    @property
    def name(self) -> str:
        return "t2splat"

    @property
    def device(self) -> torch.device:
        vertices = getattr(self._model, "vertices", None)
        if isinstance(vertices, torch.Tensor):
            return vertices.device
        return torch.device("cpu")

    def backend_status(self) -> dict[str, Any]:
        return self._backend.status(self.repo_root)

    def configure(self, **kwargs: Any) -> None:
        """Set render-time options: ``render_scaling`` (alias ``scaling``) and ``bg_color``."""
        for key, value in kwargs.items():
            if key in {"render_scaling", "scaling"}:
                self._render_scaling = int(value)
                if self._model is not None:
                    self._model.scaling = self._render_scaling
            elif key == "bg_color":
                name = str(value).lower()
                if name in {"white", "1", "1.0"}:
                    self._background_color = [1.0, 1.0, 1.0]
                elif name in {"black", "0", "0.0"}:
                    self._background_color = [0.0, 0.0, 0.0]
                else:
                    raise ValueError("T2SplatAdapter bg_color must be 'white' or 'black'.")
                self._background_color_override = list(self._background_color)
            else:
                raise KeyError(f"Unknown T2SplatAdapter config key: {key!r}")

    def _ensure_imports(self) -> None:
        if self._initialized:
            return
        try:
            from msbench.vendor.t2splat.scene.triangle_model import TriangleModel
            from msbench.vendor.t2splat.triangle_renderer import render
            from msbench.vendor.t2splat.utils.graphics_utils import getProjectionMatrix, getWorld2View2
        except ImportError as exc:
            raise ImportError(
                "Cannot import the bundled T²-Splat renderer. Build it with "
                "MSBENCH_CUDA_BACKENDS=t2splat,simple-knn and install effrdel."
            ) from exc
        self._TriangleModel = TriangleModel
        self._render = render
        self._getWorld2View2 = getWorld2View2
        self._getProjectionMatrix = getProjectionMatrix
        self._initialized = True

    def load_checkpoint(self, path: str) -> None:
        """Load a ``point_cloud/iteration_N`` directory or its ``point_cloud_state_dict.pt``."""
        self._ensure_imports()
        ckpt_path = Path(path)
        ckpt_dir = ckpt_path.parent if ckpt_path.is_file() else ckpt_path
        state_path = ckpt_dir / "point_cloud_state_dict.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {state_path}")

        state = torch.load(str(state_path), map_location="cpu", weights_only=False)
        max_sh_degree = int(math.sqrt(state["features_rest"].shape[1] + 1)) - 1
        del state
        model = self._TriangleModel(max_sh_degree)
        # Restores face_textures / face_tex_res / use_mip_texture when present.
        model.load_parameters(str(ckpt_dir), device="cuda")
        model.scaling = self._render_scaling

        self._model = model
        self._checkpoint_path = str(state_path)
        self._checkpoint_dir = ckpt_dir
        match = re.fullmatch(r"iteration_(\d+)", ckpt_dir.name)
        if match and ckpt_dir.parent.name == "point_cloud":
            self._model_path, self._loaded_iteration = ckpt_dir.parent.parent, int(match.group(1))
        else:
            self._model_path, self._loaded_iteration = None, None
        if self._background_color_override is None:
            self._background_color = self._cfg_background()
        else:
            self._background_color = list(self._background_color_override)

    def _cfg_background(self) -> list[float]:
        cfg_args = self._model_path / "cfg_args" if self._model_path is not None else None
        if cfg_args is not None and cfg_args.exists():
            match = re.search(r"white_background=(True|False)", cfg_args.read_text(encoding="utf-8"))
            if match and match.group(1) == "True":
                return [1.0, 1.0, 1.0]
        return [0.0, 0.0, 0.0]

    def load_scene(self, dataset_path: str, split: str = "test") -> None:
        raise NotImplementedError("T2SplatAdapter renders MeshSuite CameraBatch inputs directly.")

    def _require_model(self):
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")
        return self._model

    def model_stats(self) -> dict[str, Any]:
        model = self._require_model()
        vertices = model.get_vertices.detach().cpu()
        faces = model.get_triangle_indices.detach().cpu().long()
        vertex_weight = model.get_vertex_weight.detach().cpu().squeeze()
        tri_stats = compute_triangle_stats(vertices[faces], vertex_weight[faces].mean(dim=1))
        face_textures = getattr(model, "face_textures", None)
        params = [model.vertices, model.vertex_weight, model._features_dc, model._features_rest, face_textures]
        num_params = sum(p.numel() for p in params if isinstance(p, torch.Tensor))
        ckpt_size_mb = 0.0
        if self._checkpoint_path and os.path.exists(self._checkpoint_path):
            ckpt_size_mb = os.path.getsize(self._checkpoint_path) / (1024 * 1024)
        weights = vertex_weight.numpy()
        return {
            "method": "t2splat",
            "scene": "",
            "primitive_count": int(faces.shape[0]),
            "vertex_count": int(vertices.shape[0]),
            "face_count": int(faces.shape[0]),
            "trainable_param_count": int(num_params),
            "checkpoint_size_mb": round(ckpt_size_mb, 2),
            "triangle_area_mean": tri_stats.get("triangle_area_mean", 0.0),
            "triangle_area_median": tri_stats.get("triangle_area_median", 0.0),
            "triangle_area_p95": tri_stats.get("triangle_area_p95", 0.0),
            "opacity_mean": tri_stats.get("opacity_mean", 0.0),
            "opacity_median": tri_stats.get("opacity_median", 0.0),
            "opacity_histogram": tri_stats.get("opacity_histogram", []),
            "backend_extra": {
                "sigma": float(model.get_sigma),
                "active_sh_degree": int(model.active_sh_degree),
                "max_sh_degree": int(model.max_sh_degree),
                "face_tex_res": int(getattr(model, "face_tex_res", 0)) if face_textures is not None else 0,
                "use_mip_texture": bool(getattr(model, "use_mip_texture", False)),
                "vertex_weight_mean": float(np.mean(weights)),
                "vertex_weight_min": float(np.min(weights)),
                "vertex_weight_max": float(np.max(weights)),
            },
        }

    def _build_native_camera(self, cameras: CameraBatch, idx: int = 0):
        viewmat = cameras.viewmats[idx].cpu().numpy()
        R = viewmat[:3, :3].T
        T = viewmat[:3, 3]
        K = cameras.Ks[idx]
        fovX = 2.0 * math.atan(cameras.width / (2.0 * K[0, 0].item()))
        fovY = 2.0 * math.atan(cameras.height / (2.0 * K[1, 1].item()))
        wvt = torch.tensor(self._getWorld2View2(R, T), dtype=torch.float32).transpose(0, 1).cuda()
        proj = self._getProjectionMatrix(
            znear=cameras.near, zfar=cameras.far, fovX=fovX, fovY=fovY
        ).transpose(0, 1).cuda()
        full_proj = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        return SimpleNamespace(
            FoVx=fovX,
            FoVy=fovY,
            image_height=cameras.height,
            image_width=cameras.width,
            world_view_transform=wvt,
            full_proj_transform=full_proj,
            camera_center=wvt.inverse()[3, :3],
        )

    def render(self, cameras: CameraBatch, *, mode: str = "eval") -> RenderOutput:
        model = self._require_model()
        self._ensure_imports()
        native_cam = self._build_native_camera(cameras, idx=0)
        pipe = SimpleNamespace(debug=False, convert_SHs_python=False)
        bg_color = torch.tensor(self._background_color, dtype=torch.float32, device="cuda")
        grad_ctx = torch.no_grad() if mode == "eval" else torch.enable_grad()
        with grad_ctx:
            rendering = self._render(native_cam, model, pipe, bg_color)
        return RenderOutput(
            rgb=rendering["render"].permute(1, 2, 0).contiguous(),
            alpha=rendering["rend_alpha"].squeeze(0) if "rend_alpha" in rendering else None,
            depth=rendering["surf_depth"].squeeze(0) if "surf_depth" in rendering else None,
            normal=rendering["rend_normal"].permute(1, 2, 0).contiguous() if "rend_normal" in rendering else None,
            radii=rendering.get("radii"),
            visibility=rendering.get("visibility_filter"),
            extras={
                "background_color": list(self._background_color),
                "scaling": rendering.get("scaling"),
                "max_blending": rendering.get("max_blending"),
                "triangle_was_rendered": rendering.get("triangle_was_rendered"),
                "surf_normal": rendering.get("surf_normal"),
            },
        )

    def export_mesh(
        self,
        path: str | Path,
        *,
        clean: bool = False,
        min_faces_ratio: float = 0.01,
        **_ignored: Any,
    ) -> Path:
        """Write the trained mesh with SH-DC vertex colours.

        ``clean`` (``mesh.clean`` in the config) drops connected components with
        fewer than ``min_faces_ratio`` of the largest component's faces, as upstream
        ``create_ply.py --clean --geometry-only`` does before its DTU evaluation. TSDF options passed by the generic mesh
        CLI are ignored: T²-Splat outputs its mesh directly.
        """
        from msbench.core.mesh_eval import export_ply

        self._require_model()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        primitive = self.to_primitive()
        vertices = primitive.vertices.detach().cpu().numpy()
        faces = primitive.faces.detach().cpu().long().numpy()
        sh_dc = primitive.sh_coeffs.detach().cpu().numpy()
        # Upstream bake_vertex_colors(mode="dc"): SH_C0 * dc + 0.5, rounded to uint8.
        colors = np.floor(np.clip(sh_dc * 0.28209479177387814 + 0.5, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
        counts = {"vertices_before_clean": int(len(vertices)), "faces_before_clean": int(len(faces))}
        if clean:
            mesh = clean_t2splat_mesh(vertices, faces, colors, min_faces_ratio=min_faces_ratio)
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.uint8)
        export_ply(vertices, faces, path, colors=colors)
        metadata = {
            "msbench_mesh_export": {
                "method": "t2splat",
                "pipeline": "native_mesh",
                "version": MESH_EXPORT_VERSION,
                "checkpoint": str(self._checkpoint_dir) if self._checkpoint_dir is not None else None,
                "iteration": self._loaded_iteration,
                "clean": bool(clean),
                "min_faces_ratio": float(min_faces_ratio),
                **counts,
                "vertices": int(len(vertices)),
                "faces": int(len(faces)),
            }
        }
        path.with_suffix(path.suffix + ".msbench.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return path

    def to_primitive(self) -> Any:
        from msbench.primitives.mesh_triangle import IndexedMeshTriangle

        model = self._require_model()
        vertices = model.get_vertices.detach().clone()
        faces = model.get_triangle_indices.detach().clone().long()
        vertex_weight = model.get_vertex_weight.detach().clone().squeeze()
        return IndexedMeshTriangle(
            vertices=vertices,
            faces=faces,
            vertex_weights=vertex_weight,
            opacity=vertex_weight[faces].amin(dim=1),
            sigma=model.get_sigma,
            # View-independent DC colour; the full SH/MipTex model is used for rendering.
            sh_coeffs=model._features_dc.detach().reshape(vertices.shape[0], 3).clone(),
        )
