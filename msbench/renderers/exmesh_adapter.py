"""Native ExMesh adapter with lazy optional renderer imports."""

from __future__ import annotations

from pathlib import Path

import torch

from msbench.core.registry import register
from msbench.renderers.base import RendererAdapter, RenderOutput
from msbench.renderers.backends import get_backend
from msbench.vendor.exmesh.runtime import native_camera, validate_model_state


@register("exmesh")
class ExMeshAdapter(RendererAdapter):
    def __init__(self, device: str = "cuda"):
        self._runtime_device = torch.device(device)
        self._checkpoint_path = None
        self._checkpoint = None
        self._model = None
        self._context = None
        self._background = [0.0, 0.0, 0.0]

    @property
    def name(self):
        return "exmesh"

    @property
    def device(self):
        return self._runtime_device

    def backend_status(self):
        return get_backend(self.name).status()

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if key != "bg_color":
                raise KeyError(f"Unknown ExMesh render option: {key}")
            if value not in {"black", "white"}:
                raise ValueError("ExMesh bg_color must be black or white")
            self._background = [float(value == "white")] * 3
            self._background_override = True

    def load_checkpoint(self, path):
        path = Path(path)
        if path.is_dir():
            candidates = [path / "final.pt", path / "model_state_dict.pt", path / "ckpt/final.pt"]
            path = next((p for p in candidates if p.is_file()), candidates[0])
        data = torch.load(path, map_location="cpu", weights_only=False)
        state = data.get("model", data)
        validate_model_state(state)
        self._checkpoint, self._checkpoint_path = data, str(path.resolve())
        self._state = state
        if not getattr(self, "_background_override", False):
            self._background = data.get("metadata", {}).get("background_color", [0.0, 0.0, 0.0])
        self._model, self._context = None, None

    def load_scene(self, dataset_path, split="test"):
        from msbench.core.datasets import load_dataset

        self._dataset = load_dataset(dataset_path, split=split)
        return self._dataset

    def _ensure_runtime(self):
        if self._model is not None:
            return
        if self._checkpoint is None:
            raise RuntimeError("Load an ExMesh checkpoint first")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "ExMesh rasterization requires CUDA; CPU supports checkpoint inspection/export only"
            )
        from msbench.vendor.exmesh.runtime import restore_model
        import nvdiffrast.torch as dr

        self._model = restore_model(self._state, device=self.device)
        self._context = dr.RasterizeCudaContext(device=self.device)

    def render(self, cameras, *, mode="eval"):
        self._ensure_runtime()
        from msbench.vendor.exmesh.renderer import render

        cameras = cameras.to(self.device)
        bg = torch.tensor(self._background, device=self.device)
        outputs = []
        with torch.set_grad_enabled(mode.startswith("train")):
            for i in range(cameras.batch_size):
                outputs.append(
                    render(native_camera(cameras, i), self._model, None, bg, context=self._context)
                )

        def pack(name, kind):
            values = [
                o[name].permute(1, 2, 0) if kind == "rgb" else o[name].squeeze(0) for o in outputs
            ]
            return values[0] if len(values) == 1 else torch.stack(values)

        return RenderOutput(
            rgb=pack("render", "rgb"),
            alpha=pack("rend_alpha", "scalar"),
            depth=pack("rend_depth", "scalar"),
            normal=pack("rend_normal", "rgb"),
            extras={
                "background_color": self._background,
                "depth_semantics": "upstream_interpolated_vertex_distance",
                "camera_convention": "upstream_centered",
                "face_coverage": outputs[0]["scaling"],
            },
        )

    def model_stats(self):
        if self._checkpoint is None:
            raise RuntimeError("Load an ExMesh checkpoint first")
        s = self._state
        return {
            "method": self.name,
            "primitive_count": len(s["faces"]),
            "vertex_count": len(s["vertices"]),
            "face_count": len(s["faces"]),
            "trainable_param_count": s["vertices"].numel() + s["texture"].numel(),
            "checkpoint_size_mb": Path(self._checkpoint_path).stat().st_size / 2**20,
            "backend_extra": {
                "uv_vertex_count": len(s["uvs"]),
                "texture_shape": list(s["texture"].shape),
                "camera_convention": "upstream_centered",
            },
        }

    def to_primitive(self):
        from msbench.primitives.mesh_triangle import IndexedMeshTriangle

        return IndexedMeshTriangle(self._state["vertices"], self._state["faces"].long())

    def export_mesh(self, path, **kwargs):
        from msbench.core.mesh_eval import export_ply

        return export_ply(self._state["vertices"].numpy(), self._state["faces"].numpy(), path)

    def export_textured_mesh(self, path):
        """Write OBJ/MTL/PNG with independent geometry and UV indices."""
        from PIL import Image

        path = Path(path).with_suffix(".obj")
        path.parent.mkdir(parents=True, exist_ok=True)
        s = self._state
        tex = (s["texture"].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        Image.fromarray(tex[::-1]).save(path.with_suffix(".png"))
        path.with_suffix(".mtl").write_text(
            f"newmtl appearance\nKd 1 1 1\nmap_Kd {path.stem}.png\n"
        )
        with path.open("w") as f:
            f.write(f"mtllib {path.stem}.mtl\nusemtl appearance\n")
            for v in s["vertices"].tolist():
                f.write("v " + " ".join(map(str, v)) + "\n")
            for uv in s["uvs"].tolist():
                f.write("vt " + " ".join(map(str, uv)) + "\n")
            for face, uvface in zip(s["faces"].tolist(), s["uv_indices"].tolist()):
                f.write("f " + " ".join(f"{v + 1}/{uv + 1}" for v, uv in zip(face, uvface)) + "\n")
        return path
