"""ExMesh reference training semantics exposed through MeshSuite's hooks."""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from msbench.core.datasets import load_dataset
from msbench.renderers.base import RenderOutput
from msbench.trainers.hooks import TrainingMethod
from msbench.trainers.registry import register_training_method


DEFAULTS = dict(
    position_lr_delay_mult=0.01,
    position_lr_max_steps=10000,
    feature_lr=0.0025,
    lambda_dssim=0.2,
    densification_interval=500,
    split_from_iter=1000,
    split_until_iter=7000,
    random_background=False,
    lambda_smooth=3000.0,
    lambda_deviation=10000.0,
    lambda_silhouette=0.005,
    lambda_depth=0.1,
    lr_vertices_init=0.0005,
    split_size=24.0,
    max_shapes=200000,
    add_shape=1.2,
    ema_alpha=0.95,
    degeneracy_threshold=0.05,
    texture_resolution=[2560, 2560],
    smooth_ref_ema=0.99,
)


@register_training_method("exmesh")
class ExMeshTrainingMethod(TrainingMethod):
    def __init__(
        self,
        dataset,
        output_dir,
        *,
        max_steps=10000,
        dataset_type="colmap",
        images="images",
        resolution=1,
        eval_split=False,
        llffhold=8,
        white_background=False,
        dtu_eval_mode="full",
        initial_mesh=None,
        depth_dir=None,
        seed=0,
        topology_enabled=True,
        resume=False,
        foreground_training=False,
        extra_args=None,
    ):
        if foreground_training:
            raise ValueError(
                "ExMesh uses full composited RGB plus a silhouette loss; foreground-only RGB is unsupported"
            )
        self.dataset_path = Path(dataset)
        self.output_dir = Path(output_dir)
        self.max_steps = max_steps
        self.dataset_type = dataset_type
        self.images, self.resolution = images, resolution
        self.eval_split, self.llffhold = eval_split, llffhold
        self.background = float(white_background)
        self.dtu_eval_mode = dtu_eval_mode
        self.initial_mesh = Path(initial_mesh) if initial_mesh else self.dataset_path / "mesh.ply"
        self.depth_dir = Path(depth_dir) if depth_dir else self.dataset_path / "mono_priors/da3"
        self.seed, self.topology_enabled, self.resume = seed, topology_enabled, resume
        unknown = set(extra_args or {}) - DEFAULTS.keys()
        if unknown:
            raise ValueError(f"Unknown ExMesh training options: {sorted(unknown)}")
        self.opt = SimpleNamespace(**{**DEFAULTS, **(extra_args or {})})
        if (
            not isinstance(self.opt.texture_resolution, (list, tuple))
            or len(self.opt.texture_resolution) != 2
        ):
            raise ValueError("ExMesh texture_resolution must be a YAML list [height, width]")
        self._initialized = False
        self._step = 0
        self._save_requested = False
        self._pool = []

    def _ensure_initialized(self):
        if self._initialized:
            return
        from msbench.vendor.exmesh.runtime import initialize_mesh
        import nvdiffrast.torch as dr

        if not torch.cuda.is_available():
            raise RuntimeError("ExMesh training requires CUDA")
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        kwargs = dict(split="train" if self.eval_split else "all", resolution=self.resolution)
        if self.dataset_type in {"colmap", "dtu"}:
            kwargs["eval_every"] = self.llffhold
        if self.dataset_type == "colmap":
            kwargs.update(
                image_dir=self.images,
                background="white" if self.background else "black",
                use_alpha=True,
            )
        if self.dataset_type == "dtu":
            kwargs["dtu_eval_mode"] = self.dtu_eval_mode
        if self.dataset_type in {"synthetic", "nerf_synthetic"}:
            # Synthetic training must never append its test split.
            kwargs.update(split="train", background="white" if self.background else "black")
        self.dataset = load_dataset(self.dataset_path, dataset_type=self.dataset_type, **kwargs)
        if len(self.dataset) == 0:
            raise ValueError("No ExMesh training views")
        self._order = list(range(len(self.dataset)))
        random.shuffle(self._order)
        self._pool = self._order.copy()
        self._targets, self._masks, self._depths = {}, {}, {}
        for index in self._order:
            sample = self.dataset.sample(index)
            size = (sample.camera.width, sample.camera.height)
            # Reference loadCam resizes each image channel separately using PIL defaults.
            with Image.open(sample.image_path) as im:
                channels = im.convert("RGBA").split()
                rgb = np.stack([np.asarray(c.resize(size)) for c in channels[:3]], axis=-1).copy()
                if "A" in im.getbands():
                    mask = np.asarray(im.split()[3].resize(size)).copy() / 255.0
                else:
                    mask_path = self.dataset_path / "train_mask" / f"{sample.name}_gtmask.png"
                    if not mask_path.exists():
                        raise FileNotFoundError(f"ExMesh needs a silhouette mask: {mask_path}")
                    with Image.open(mask_path) as m:
                        mask = np.asarray(m.convert("L").resize(size)).copy() / 255.0
            mask = torch.as_tensor(mask, dtype=torch.float32, device="cuda")
            image = torch.as_tensor(rgb, dtype=torch.float32, device="cuda") / 255.0
            self._targets[index] = image * mask[..., None] + self.background * (1 - mask[..., None])
            self._masks[index] = mask
            if self.opt.lambda_depth:
                path = self.depth_dir / f"{sample.name}.npz"
                if not path.is_file():
                    raise FileNotFoundError(f"ExMesh DA3 depth prior missing: {path}")
                with np.load(path) as data:
                    depth = torch.from_numpy(data["depth"].copy()).float()
                if depth.ndim != 2 or not torch.isfinite(depth).all():
                    raise ValueError(f"Invalid DA3 depth: {path}")
                depth = F.interpolate(
                    depth[None, None], size=size[::-1], mode="bilinear", align_corners=False
                )[0, 0]
                self._depths[index] = depth.cuda()
        self.model = initialize_mesh(
            self.initial_mesh, self.opt.texture_resolution, self.background
        )
        self.model.training_setup(self.opt, self.opt.feature_lr, self.opt.lr_vertices_init)
        self.context = dr.RasterizeCudaContext()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "ckpt").mkdir(exist_ok=True)
        self._initialized = True

    def sample_batch(self):
        self._ensure_initialized()
        self.model.update_learning_rate(self._step)
        if not self._pool:
            self._pool = self._order.copy()
        self._current = self._pool.pop(random.randint(0, len(self._pool) - 1))
        sample = self.dataset.sample(self._current)
        return sample.camera.to("cuda"), self._targets[self._current]

    def render_train(self, cameras):
        from msbench.vendor.exmesh.renderer import render
        from msbench.vendor.exmesh.runtime import native_camera

        bg = (
            torch.rand(3, device="cuda")
            if self.opt.random_background
            else torch.full((3,), self.background, device="cuda")
        )
        pkg = render(native_camera(cameras), self.model, None, bg, context=self.context)
        with torch.no_grad():
            mask = pkg["scaling"].detach() > self.model.image_size
            self.model.image_size[mask] = pkg["scaling"].detach()[mask]
        return RenderOutput(
            rgb=pkg["render"].permute(1, 2, 0),
            alpha=pkg["rend_alpha"][0],
            depth=pkg["rend_depth"][0],
            normal=pkg["rend_normal"].permute(1, 2, 0),
        )

    def compute_loss(self, output, gt_images):
        from msbench.vendor.exmesh.losses import (
            l1_loss,
            ssim,
            silhouette_bce_loss,
            pearson_correlation_loss,
            laplacian_smooth_loss,
            double_vertex_deviation_loss,
        )

        opt, m = self.opt, self.model
        image, target = output.rgb.permute(2, 0, 1), gt_images.permute(2, 0, 1)
        rgb = (1 - opt.lambda_dssim) * l1_loss(image, target) + opt.lambda_dssim * (
            1 - ssim(image, target)
        )
        mask = self._masks[self._current]
        silhouette = silhouette_bce_loss(output.alpha, mask) * opt.lambda_silhouette
        if self._step < opt.split_from_iter:
            silhouette.backward(retain_graph=True)
            with torch.no_grad():
                m.mask_grad_ema += torch.linalg.vector_norm(m._vertices.grad, dim=1)
            m._vertices.grad = None
            m._texture.grad = None
        depth = image.sum() * 0.0
        if opt.lambda_depth:
            valid = (mask > 0) & (output.depth > 0)
            if not valid.any():
                raise RuntimeError(
                    "No valid rendered foreground depth; inspect camera alignment before training"
                )
            depth, _ = pearson_correlation_loss(output.depth, self._depths[self._current], valid)
            depth = depth * opt.lambda_depth
        smooth = laplacian_smooth_loss(m.get_vertices, m.get_faces) * opt.lambda_smooth
        deviation = (
            double_vertex_deviation_loss(m.get_vertices, m.vertices_ref, m.get_faces)
            * opt.lambda_deviation
        )
        total = rgb + silhouette + depth + smooth + deviation
        if not torch.isfinite(total):
            raise RuntimeError(f"Nonfinite ExMesh loss at step {self._step}")
        return dict(
            total=total,
            image=rgb,
            silhouette=silhouette,
            depth=depth,
            smooth=smooth,
            deviation=deviation,
        )

    @torch.no_grad()
    def update_structure(self, step):
        if not self.topology_enabled:
            return None
        from msbench.vendor.exmesh.geometry import uvmap_to_vertex_color

        opt, m, bg = self.opt, self.model, self.background
        events = []

        def colors():
            m._vertices_color = torch.as_tensor(
                uvmap_to_vertex_color(
                    m._vertices, m._uvs, m._texture, m._vmapping, m._texture_mask
                ),
                dtype=torch.float32,
                device="cuda",
            )

        def boundary(rounds):
            dead = m.compute_boundary_dead_mask(
                grad_percent=0.2,
                area_percent=0.2,
                white_thresh=0.995,
                grad_thresh=1e-8,
                bg_color=bg,
            )
            m.prune_boundary_vertices(dead, n_rounds=rounds)

        if step == 1:
            colors()
            threshold = torch.quantile(m.compute_face_areas(), 0.1)
            for _ in range(10):
                dead = (m.compute_degeneracy_ratio() < 0.1) | (m.compute_face_areas() < threshold)
                m.merge_close_vertex(dead, length_ratio_thresh=0.2)
            m.recreate_uvmap(bg_color=bg)
            m._reset_optimizer()
            m.vertices_ref = m.get_vertices.detach().clone()
            m.mask_grad_ema = torch.zeros(len(m._vertices), device="cuda")
            events.append("initial_merge")
        if step % opt.densification_interval == 0 and step < opt.split_from_iter:
            colors()
            boundary(5)
            m.vertices_ref = m.get_vertices.detach().clone()
            m.mask_grad_ema = torch.zeros(len(m._vertices), device="cuda")
            events.append("boundary_prune")
        elif (
            opt.split_from_iter <= step <= opt.split_until_iter
            and step % opt.densification_interval == 0
        ):
            colors()
            if step == 1000:
                boundary(5)
            dead = m.compute_dead_mask(
                image_size_thresh=10,
                degeneracy_thresh=opt.degeneracy_threshold * 2,
                area_percent=0.5,
            )
            m.add_new_face(cap_max=opt.max_shapes, dead_mask=dead)
            if step % (opt.densification_interval * 4) == 0:
                boundary(2)
                for _ in range(5):
                    dead = m.compute_dead_mask(
                        image_size_thresh=0,
                        degeneracy_thresh=opt.degeneracy_threshold,
                        area_percent=0.3,
                    )
                    m.merge_close_vertex(dead, length_ratio_thresh=0.2)
            m.image_size = torch.zeros(len(m._faces), device="cuda")
            m.vertices_ref = m.get_vertices.detach().clone()
            events.append("split_merge")
        if step in (500, 2000, 4000, 6000):
            m.recreate_uvmap(bg_color=bg)
            m._reset_optimizer()
            events.append("recreate_uv")
        if events:
            info = dict(
                step=step, type="+".join(events), vertices=len(m._vertices), faces=len(m._faces)
            )
            with (self.output_dir / "topology.jsonl").open("a") as f:
                f.write(json.dumps(info) + "\n")
            return info

    def optimizer_step(self):
        m = self.model
        if self._step < self.max_steps:
            m.optimizer.step()
            m.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                beta = self.opt.smooth_ref_ema
                m.vertices_ref.mul_(beta).add_((1 - beta) * m.get_vertices.detach())
        if self._save_requested:
            self.save_checkpoint(self.output_dir / "ckpt" / f"{self._step}.pt")
            self._save_requested = False
        if self._step == self.max_steps:
            self.save_checkpoint(self.output_dir / "ckpt/final.pt", inference_only=True)
            mesh_dir = self.output_dir / "mesh"
            mesh_dir.mkdir(exist_ok=True)
            m.save_as_ply(str(mesh_dir / "mesh.ply"))

    def on_epoch_end(self, epoch):
        # The common loop requests saving before topology/optimizer updates.
        # Commit resume snapshots only after that iteration has fully completed.
        self._save_requested = True

    def on_step_end(self, step, losses):
        with (self.output_dir / "losses.jsonl").open("a") as f:
            f.write(json.dumps(dict(step=step, **losses)) + "\n")

    def get_lr(self):
        return {g["name"]: g["lr"] for g in self.model.optimizer.param_groups}

    def save_checkpoint(self, path, *, inference_only=False):
        from msbench.vendor.exmesh.runtime import model_state, STAT_FIELDS, validate_model_state

        state = model_state(self.model)
        validate_model_state(state)
        data = dict(
            schema_version=1,
            method="exmesh",
            checkpoint_kind="inference" if inference_only else "training",
            step=self._step,
            model=state,
            metadata=dict(
                background_color=[self.background] * 3,
                camera_convention="upstream_centered",
                dataset_type=self.dataset_type,
                training_views=[self.dataset.sample(i).name for i in self._order],
            ),
        )
        if not inference_only:
            data.update(
                statistics={k: getattr(self.model, k).detach().cpu() for k in STAT_FIELDS},
                optimizer=self.model.optimizer.state_dict(),
                options=vars(self.opt),
                sampler=dict(order=self._order, pool=self._pool),
                rng=dict(
                    python=random.getstate(),
                    numpy=np.random.get_state(),
                    torch=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state_all(),
                ),
            )
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        torch.save(data, tmp)
        tmp.replace(path)

    def resume_from_checkpoint(self, output_dir, max_steps):
        self._ensure_initialized()
        if not self.resume:
            return 0
        from msbench.vendor.exmesh.runtime import restore_model

        ckpt_dir = self.output_dir / "ckpt"
        paths = list(ckpt_dir.glob("[0-9]*.pt"))
        if not paths:
            return 0
        path = max(paths, key=lambda p: int(p.stem))
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data["options"] != vars(self.opt):
            raise ValueError("ExMesh resume options differ from the saved training state")
        self.model = restore_model(data["model"])
        self.model.training_setup(self.opt, self.opt.feature_lr, self.opt.lr_vertices_init)
        self.model.optimizer.load_state_dict(data["optimizer"])
        for k, v in data["statistics"].items():
            setattr(self.model, k, v.cuda())
        self._order, self._pool = data["sampler"]["order"], data["sampler"]["pool"]
        random.setstate(data["rng"]["python"])
        np.random.set_state(data["rng"]["numpy"])
        torch.set_rng_state(data["rng"]["torch"])
        torch.cuda.set_rng_state_all(data["rng"]["cuda"])
        self._step = int(data["step"])
        return self._step
