"""ExMesh contract, UV topology and real CUDA regression checks."""

import pytest
import torch

from msbench.renderers.exmesh_adapter import ExMeshAdapter
from msbench.vendor.exmesh.runtime import validate_model_state


def quad_state():
    return dict(
        vertices=torch.tensor([[-0.6, -0.4, 0], [0.6, -0.4, 0], [0.6, 0.4, 0], [-0.6, 0.4, 0]]),
        faces=torch.tensor([[0, 1, 2], [0, 2, 3]]),
        uvs=torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        uv_indices=torch.tensor([[0, 1, 2], [0, 2, 3]]),
        vmapping=torch.arange(4),
        texture=torch.rand(3, 32, 32),
        texture_mask=torch.ones(32, 32),
    )


def test_exmesh_cpu_checkpoint_inspection_and_uv_validation(tmp_path):
    state = quad_state()
    validate_model_state(state)
    path = tmp_path / "model.pt"
    torch.save(dict(model=state, metadata=dict(background_color=[0.0, 0.0, 0.0])), path)
    adapter = ExMeshAdapter(device="cpu")
    adapter.load_checkpoint(path)
    assert adapter.model_stats()["primitive_count"] == 2
    assert adapter.model_stats()["trainable_param_count"] == 4 * 3 + 3 * 32 * 32
    assert adapter.to_primitive().faces.shape == (2, 3)
    obj = adapter.export_textured_mesh(tmp_path / "mesh.obj")
    assert obj.is_file() and obj.with_suffix(".png").is_file() and obj.with_suffix(".mtl").is_file()
    state["vmapping"] = torch.tensor([1, 0, 2, 3])
    with pytest.raises(ValueError, match="topology disagree"):
        validate_model_state(state)


def test_exmesh_textured_export_cli(tmp_path):
    import numpy as np
    from PIL import Image
    from typer.testing import CliRunner
    from msbench.cli.main import app

    path = tmp_path / "model.pt"
    state = quad_state()
    state.update(
        uvs=state["uvs"][torch.tensor([0, 1, 2, 0, 2, 3])],
        vmapping=torch.tensor([0, 1, 2, 0, 2, 3]),
        uv_indices=torch.tensor([[0, 1, 2], [3, 4, 5]]),
    )
    state["texture"] = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]]
    )
    state["texture_mask"] = torch.ones(2, 2)
    torch.save(state, path)
    output = tmp_path / "mesh.obj"
    result = CliRunner().invoke(
        app, ["export", "textured-mesh", "-m", "exmesh", "-c", str(path), "-o", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert output.is_file() and output.with_suffix(".png").is_file()
    assert "f 1/4 3/5 4/6" in output.read_text()
    pixels = np.asarray(Image.open(output.with_suffix(".png")))
    assert pixels[0, 0].tolist() == [0, 0, 255]  # PNG top row is the UV v=1 edge.
    assert pixels[1, 0].tolist() == [255, 0, 0]


def test_colmap_alpha_compositing_is_opt_in(tmp_path):
    from PIL import Image
    from msbench.core.datasets import load_dataset

    (tmp_path / "sparse/0").mkdir(parents=True)
    (tmp_path / "images").mkdir()
    (tmp_path / "sparse/0/cameras.txt").write_text("1 PINHOLE 16 16 10 10 8 8\n")
    (tmp_path / "sparse/0/images.txt").write_text("1 1 0 0 0 0 0 2 1 frame.png\n\n")
    Image.new("RGBA", (16, 16), (50, 100, 150, 0)).save(tmp_path / "images/frame.png")
    raw = load_dataset(tmp_path, dataset_type="colmap", split="all").sample(0)
    composited = load_dataset(
        tmp_path, dataset_type="colmap", split="all", background="black"
    ).sample(0)
    assert raw.mask is None and raw.image[0, 0, 2] > 0.5
    assert composited.mask is not None and composited.image.sum() == 0
    assert composited.metadata["eval_background_color"] == (0.0, 0.0, 0.0)
    white = load_dataset(
        tmp_path, dataset_type="colmap", split="all", background="white"
    ).sample(0)
    assert torch.all(white.image == 1) and white.mask.sum() == 0
    assert white.metadata["eval_background_color"] == (1.0, 1.0, 1.0)


def test_2dgs_staging_resizes_images_and_intrinsics_without_changing_poses(tmp_path):
    import json
    from PIL import Image
    from msbench.core.datasets import load_dataset, _resolve_resolution
    from msbench.trainers.exmesh_preprocess import stage_colmap

    source, target = tmp_path / "source", tmp_path / "target"
    (source / "sparse/0").mkdir(parents=True)
    (source / "images").mkdir()
    (source / "sparse/0/cameras.txt").write_text("7 PINHOLE 16 12 10 11 8 6\n")
    (source / "sparse/0/images.txt").write_text("100 1 0 0 0 0.3 -0.7 2.4 7 frame.png\n\n")
    (source / "sparse/0/points3D.txt").write_text("1 0 0 0 100 120 140 0\n")
    Image.new("RGBA", (16, 12), (50, 100, 150, 255)).save(source / "images/frame.png")
    before = (source / "sparse/0/cameras.txt").read_bytes()
    report = stage_colmap(source, target, [8, 6])
    actual = load_dataset(target, dataset_type="colmap", split="all").sample(0)
    expected = load_dataset(source, dataset_type="colmap", split="all", resolution=[8, 6]).sample(0)
    torch.testing.assert_close(actual.camera.viewmats, expected.camera.viewmats)
    torch.testing.assert_close(actual.camera.Ks, expected.camera.Ks)
    assert (actual.camera.width, actual.camera.height) == (8, 6)
    assert report["views"][0]["intrinsics"] == [5, 5.5, 4, 3]
    assert (source / "sparse/0/cameras.txt").read_bytes() == before
    assert json.loads((target / "input_manifest.json").read_text())["split"] == "all"
    assert (target / "sparse/0/points3D.txt").is_symlink()
    # 2DGS's COLMAP loader ignores alpha when choosing the RGB training target.
    # Undefined RGB in transparent pixels must be replaced, with alpha retained.
    import numpy as np
    rgba = np.zeros((12, 16, 4), dtype=np.uint8)
    rgba[..., :3] = [20, 80, 140]
    rgba[:, 5:10, 3] = 128
    rgba[:, 10:, 3] = 255
    Image.fromarray(rgba).save(source / "images/frame.png")
    staged = tmp_path / "white"
    manifest = stage_colmap(source, staged, [16, 12], background="white")
    image = np.asarray(Image.open(staged / "images/frame.png"))
    np.testing.assert_array_equal(image[..., 3], rgba[..., 3])
    assert np.all(image[:, :5, :3] == 255)
    np.testing.assert_array_equal(image[:, 10:, :3], rgba[:, 10:, :3])
    np.testing.assert_array_equal(image[0, 7, :3], [137, 167, 197])
    assert manifest["background"] == "white"
    for invalid in ([8, 0], [8], [8, 6, 2], [8.0, 6]):
        with pytest.raises(ValueError, match="Explicit resolution"):
            _resolve_resolution(16, 12, invalid)


def test_exmesh_dtu_configs_use_shared_protocol_with_published_parameters():
    from msbench.core.config import Config
    from msbench.core.builder import build_training_method
    from msbench.cli.train import inherit_dataset_defaults
    from msbench.trainers.exmesh_preprocess import TWO_DGS_REVISION

    expected_params = {
        24: {"lambda_smooth": 3000, "lambda_depth": 0.1},
        37: {"lambda_smooth": 3000, "lambda_depth": 0.1},
        40: {"lambda_smooth": 3000, "lambda_depth": 0.03},
        55: {"lambda_smooth": 3000, "lambda_depth": 0.01},
        63: {"lambda_smooth": 3000, "lambda_depth": 0.03},
        65: {"lambda_smooth": 3000, "lambda_depth": 0.01},
        69: {"lambda_smooth": 1000, "lambda_depth": 0.1},
        83: {"lambda_smooth": 3000, "lambda_depth": 0.01},
        97: {"lambda_smooth": 3000, "lambda_depth": 0.01},
        105: {"lambda_smooth": 3000, "lambda_depth": 0.03},
        106: {"lambda_smooth": 3000, "lambda_depth": 0.1},
        110: {"lambda_smooth": 1000, "lambda_depth": 0.01},
        114: {"lambda_smooth": 3000, "lambda_depth": 0.03},
        118: {"lambda_smooth": 1000, "lambda_depth": 0.01},
        122: {"lambda_smooth": 1000, "lambda_depth": 0.03},
    }
    outputs = set()
    for sid, params in expected_params.items():
        cfg = Config.fromfile(f"configs/exmesh/dtu/scan{sid}.yaml")
        # Same dataset protocol as every other method in MeshSuite.
        other = Config.fromfile(f"configs/triangle-splatting/dtu/scan{sid}.yaml")
        assert cfg.dataset.to_dict() == other.dataset.to_dict()
        # DTU is a mesh benchmark: every view trains, nothing is held out.
        assert cfg.dataset.eval_split is False
        assert cfg.mesh.split == other.mesh.split == "all"
        assert cfg.adapter.render_params.bg_color == other.adapter.render_params.bg_color
        assert cfg.trainer.white_background == other.trainer.white_background

        trainer = inherit_dataset_defaults(cfg.trainer.to_dict(), cfg.dataset.to_dict())
        method = build_training_method(
            trainer, default_args=dict(dataset=cfg.dataset.root, output_dir=cfg.output.dir)
        )
        # ExMesh paper Sec. 4.1 schedule; per-scene weights from the official run_dtu.py.
        assert method.max_steps == 8000 and method.opt.split_until_iter == 7000
        assert method.opt.lambda_depth == params["lambda_depth"]
        assert method.opt.lambda_smooth == params["lambda_smooth"]
        assert method.opt.lambda_deviation == 10000
        assert method.opt.lr_vertices_init == 0.0005 and method.opt.feature_lr == 0.0025
        assert method.background == float(cfg.trainer.white_background)
        assert not method.eval_split and method.resolution == cfg.dataset.resolution
        # 2DGS initialization sees the same views as ExMesh training: all of them.
        assert cfg.preprocess.type == "exmesh_2dgs"
        assert cfg.preprocess.holdout == 0
        # The paper's 2DGS initializer, explicitly pinned to the official 2DGS source.
        assert cfg.preprocess.source_repo == "outputs/exmesh-preprocess/2dgs_source"
        assert cfg.preprocess.source_revision == TWO_DGS_REVISION
        assert str(method.initial_mesh).startswith(cfg.preprocess.output)
        assert cfg.output.dir == other.output.dir.replace("triangle-splatting", "exmesh")
        assert cfg.mesh.output == cfg.output.mesh_file == cfg.output.dir + "/mesh/mesh.ply"
        assert cfg.adapter.checkpoint == cfg.output.dir + "/ckpt/final.pt"
        outputs.add(cfg.output.dir)
    assert len(outputs) == 15


def test_synthetic_pgsr_stage_preserves_cameras_and_excludes_test(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from plyfile import PlyData, PlyElement
    from msbench.core.datasets import load_dataset

    spec = importlib.util.spec_from_file_location(
        "exmesh_synthetic_stage", Path("tools/prepare_exmesh_synthetic.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, target = tmp_path / "source", tmp_path / "stage"
    (source / "train").mkdir(parents=True)
    Image.new("RGBA", (16, 12), (20, 60, 100, 0)).save(source / "train/r_0.png")
    pose = np.eye(4)
    pose[:3, :3] = [[0.8, 0, 0.6], [0, 1, 0], [-0.6, 0, 0.8]]
    pose[:3, 3] = [0.3, -0.7, 2.4]
    meta = dict(
        camera_angle_x=0.7, frames=[dict(file_path="train/r_0", transform_matrix=pose.tolist())]
    )
    (source / "transforms_train.json").write_text(json.dumps(meta))
    (source / "transforms_test.json").write_text("must never be read")
    points = np.array(
        [(0.0, 0.0, 0.0, 100, 120, 140)],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    PlyData([PlyElement.describe(points, "vertex")]).write(source / "points3d.ply")
    report = module.stage_colmap(source, target)
    actual = load_dataset(target, dataset_type="colmap", split="all").sample(0)
    expected = load_dataset(source, dataset_type="synthetic", split="train").sample(0)
    torch.testing.assert_close(actual.camera.viewmats, expected.camera.viewmats)
    torch.testing.assert_close(actual.camera.Ks, expected.camera.Ks)
    assert actual.image.min() == 1
    assert report["train_count"] == 1 and not report["test_images_used"]
    assert not report["initialization_cloud_is_ground_truth"]



def test_exmesh_synthetic_configs_follow_the_authors_setup():
    from msbench.cli.train import inherit_dataset_defaults
    from msbench.core.builder import build_training_method
    from msbench.core.config import Config
    from msbench.trainers.exmesh_preprocess import TWO_DGS_REVISION

    for scene in ("chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"):
        cfg = Config.fromfile(f"configs/exmesh/nerf_synthetic/{scene}.yaml")
        other = Config.fromfile(f"configs/triangle-splatting/nerf_synthetic/{scene}.yaml")
        assert cfg.dataset.to_dict() == other.dataset.to_dict()
        assert cfg.dataset.background == "white" and cfg.trainer.white_background
        trainer = inherit_dataset_defaults(cfg.trainer.to_dict(), cfg.dataset.to_dict())
        method = build_training_method(
            trainer, default_args=dict(dataset=cfg.dataset.root, output_dir=cfg.output.dir)
        )
        assert method.dataset_type == "synthetic" and method.eval_split
        assert method.max_steps == 8000
        # Authors: no depth supervision on NeRF-Synthetic; Table 6 regularizer weights.
        assert method.opt.lambda_depth == 0
        assert method.opt.lambda_smooth == method.opt.lambda_deviation == 1000
        # Authors: the same 2DGS 7k + 256^3 TSDF initialization as on DTU.
        assert cfg.preprocess.type == "exmesh_2dgs" and cfg.preprocess.holdout == 0
        assert cfg.preprocess.source_revision == TWO_DGS_REVISION
        assert str(method.initial_mesh).startswith(cfg.preprocess.output)
        assert cfg.output.dir == f"outputs/exmesh/nerf_synthetic/{scene}"
        assert cfg.mesh.output.startswith(cfg.output.dir)


def test_2dgs_initializer_stages_only_synthetic_training_views(tmp_path, monkeypatch):
    import json
    import numpy as np
    from PIL import Image
    from plyfile import PlyData, PlyElement
    import msbench.trainers.exmesh_preprocess as pp
    from msbench.core.config import load_config
    from msbench.core.datasets import load_dataset

    source = tmp_path / "lego"
    (source / "train").mkdir(parents=True)
    Image.new("RGBA", (16, 12), (20, 60, 100, 0)).save(source / "train/r_0.png")
    pose = np.eye(4)
    pose[:3, :3] = [[0.8, 0, 0.6], [0, 1, 0], [-0.6, 0, 0.8]]
    pose[:3, 3] = [0.3, -0.7, 2.4]
    meta = dict(
        camera_angle_x=0.7, frames=[dict(file_path="./train/r_0", transform_matrix=pose.tolist())]
    )
    (source / "transforms_train.json").write_text(json.dumps(meta))
    (source / "transforms_test.json").write_text("must never be read")
    points = np.array(
        [(0.0, 0.0, 0.0, 100, 120, 140)],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    PlyData([PlyElement.describe(points, "vertex")]).write(source / "points3d.ply")

    class Captured(Exception):
        pass

    def capture(source_repo, command, output):
        raise Captured(command)

    monkeypatch.setattr(pp.subprocess, "check_output", lambda *a, **k: pp.TWO_DGS_REVISION)
    monkeypatch.setattr(pp, "_record_effective_arguments", capture)
    cfg = load_config("configs/exmesh/nerf_synthetic/lego.yaml")
    cfg["preprocess"].update(source_repo=str(tmp_path), output=str(tmp_path / "out"))
    with pytest.raises(Captured) as captured:
        pp.prepare_exmesh_initializer(cfg, dataset_root=str(source))

    # Training views only, with the known Blender cameras and a white baked background.
    staged = tmp_path / "out/input"
    actual = load_dataset(staged, dataset_type="colmap", split="all")
    expected = load_dataset(source, dataset_type="synthetic", split="train").sample(0)
    assert len(actual) == 1
    torch.testing.assert_close(actual.sample(0).camera.viewmats, expected.camera.viewmats)
    torch.testing.assert_close(actual.sample(0).camera.Ks, expected.camera.Ks)
    assert actual.sample(0).image.min() == 1
    assert (staged / "sparse/0/points3D.txt").is_file()
    command = captured.value.args[0]
    # Official 2DGS NeRF-Synthetic flags; nothing is held out of the training split.
    assert "--eval" not in command and "--white_background" in command
    assert command[command.index("--lambda_normal") + 1] == "0.0"
    assert command[command.index("--lambda_dist") + 1] == "0.0"
    assert command[command.index("--depth_ratio") + 1] == "0.0"
    settings = json.loads((tmp_path / "out/settings.json").read_text())
    assert settings["iterations"] == 7000 and settings["mesh_resolution"] == 256


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ExMesh requires real CUDA")
def test_exmesh_cuda_render_gradients_and_roundtrip(tmp_path):
    pytest.importorskip("nvdiffrast.torch")
    from msbench.core.cameras import CameraBatch

    state = quad_state()
    path = tmp_path / "model.pt"
    torch.save(state, path)
    adapter = ExMeshAdapter()
    adapter.load_checkpoint(path)
    view = torch.eye(4)[None]
    view[0, 2, 3] = 2
    k = torch.tensor([[[50.0, 0, 32], [0, 50, 24], [0, 0, 1]]])
    camera = CameraBatch(view, torch.linalg.inv(view), k, 64, 48)
    output = adapter.render(camera, mode="train")
    assert output.rgb.shape == (48, 64, 3)
    assert output.alpha.max() > 0 and output.alpha.min() == 0
    (output.rgb.square().mean() + output.alpha.mean()).backward()
    for p in (adapter._model._vertices, adapter._model._texture):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert torch.allclose(
        output.depth[output.alpha > 0.99],
        torch.full_like(output.depth[output.alpha > 0.99], (4 + 0.6**2 + 0.4**2) ** 0.5),
        atol=1e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ExMesh requires real CUDA")
def test_exmesh_topology_split_keeps_uv_and_optimizer_binding():
    pytest.importorskip("nvdiffrast.torch")
    from types import SimpleNamespace
    from msbench.vendor.exmesh.runtime import restore_model, model_state

    state = quad_state()
    state["vertices_color"] = torch.full((4, 3), 0.5)
    model = restore_model(state)
    model.training_setup(
        SimpleNamespace(
            split_size=24, add_shape=1.2, position_lr_delay_mult=0.01, position_lr_max_steps=100
        ),
        0.0025,
        0.0005,
    )
    model._vertices.square().sum().backward()
    model.optimizer.step()
    model.optimizer.zero_grad()
    before = len(model._faces)
    # Direct operation tests UV maintenance independently of random candidate selection.
    model.sample_new_faces = lambda cap, dead: (
        torch.tensor([0], device="cuda"),
        torch.ones(2, dtype=torch.bool, device="cuda"),
    )
    model.add_new_face(20, torch.zeros(2, dtype=torch.bool, device="cuda"))
    assert len(model._faces) > before
    validate_model_state(model_state(model))
    assert model.optimizer.param_groups[0]["params"][0] is model._vertices
    assert model.optimizer.state[model._vertices]["exp_avg"].shape == model._vertices.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ExMesh requires real CUDA")
@pytest.mark.parametrize("white_background", [False, True])
def test_exmesh_resume_matches_continuous_training_across_split(tmp_path, white_background):
    pytest.importorskip("nvdiffrast.torch")
    import numpy as np
    import open3d as o3d
    from PIL import Image
    from msbench.trainers.exmesh_method import ExMeshTrainingMethod
    from msbench.trainers.loop import TrainingConfig, TrainingLoop
    from msbench.vendor.exmesh.runtime import model_state

    scene = tmp_path / "scene"
    (scene / "sparse/0").mkdir(parents=True)
    (scene / "images").mkdir()
    (scene / "mono_priors/da3").mkdir(parents=True)
    (scene / "sparse/0/cameras.txt").write_text("1 PINHOLE 64 48 50 50 32 24\n")
    text = ""
    yy, xx = np.mgrid[:48, :64]
    for i in range(3):
        name = f"{i:04d}"
        text += f"{i + 1} 1 0 0 0 {i * 0.05} 0 2 1 {name}.png\n\n"
        im = np.zeros((48, 64, 4), dtype=np.uint8)
        im[:, :, :3] = np.stack([xx * 3, yy * 4, np.full_like(xx, 90)], axis=-1)
        im[:, :, 3] = np.where((xx - 32) ** 2 + (yy - 24) ** 2 < 15**2, 255, 0)
        Image.fromarray(im).save(scene / "images" / f"{name}.png")
        np.savez(
            scene / "mono_priors/da3" / f"{name}.npz",
            depth=(2 + xx * 0.001 + yy * 0.002).astype(np.float32),
        )
    (scene / "sparse/0/images.txt").write_text(text)
    mesh = o3d.geometry.TriangleMesh.create_icosahedron(radius=0.6)
    # Avoid exact geometric ties/zero gradients, which the reference Adam eps=1e-15 amplifies.
    xyz = np.asarray(mesh.vertices).copy()
    xyz += 0.013 * np.sin(np.arange(xyz.size).reshape(xyz.shape) + 0.3)
    mesh.vertices = o3d.utility.Vector3dVector(xyz)
    mesh.paint_uniform_color([0.4, 0.5, 0.6])
    o3d.io.write_triangle_mesh(str(scene / "mesh.ply"), mesh)
    options = dict(
        texture_resolution=[32, 32],
        split_from_iter=2,
        split_until_iter=4,
        densification_interval=2,
        max_shapes=100,
    )

    def method(out, resume=False):
        return ExMeshTrainingMethod(
            scene, out, max_steps=6, seed=17, resume=resume, extra_args=options,
            white_background=white_background,
        )

    full = method(tmp_path / "full")
    TrainingLoop(
        full, TrainingConfig(max_steps=6, output_dir=str(tmp_path / "full"), save_interval=3)
    ).train()
    first = method(tmp_path / "resumed")
    TrainingLoop(
        first, TrainingConfig(max_steps=3, output_dir=str(tmp_path / "resumed"), save_interval=3)
    ).train()
    resumed = method(tmp_path / "resumed", resume=True)
    TrainingLoop(
        resumed, TrainingConfig(max_steps=6, output_dir=str(tmp_path / "resumed"), save_interval=3)
    ).train()
    a, b = model_state(full.model), model_state(resumed.model)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=1e-6, rtol=1e-5)
    assert full._pool == resumed._pool
    assert len(full.model._faces) > len(mesh.triangles)
    assert full.model.optimizer.param_groups[0]["params"][0] is full.model._vertices
    inference = torch.load(
        tmp_path / "resumed/ckpt/final.pt", map_location="cpu", weights_only=True
    )
    assert inference["checkpoint_kind"] == "inference" and "optimizer" not in inference
    training = torch.load(tmp_path / "resumed/ckpt/6.pt", map_location="cpu", weights_only=False)
    assert training["checkpoint_kind"] == "training" and "optimizer" in training
    assert inference["metadata"]["background_color"] == [float(white_background)] * 3
    for index, target in full._targets.items():
        assert torch.all(target[full._masks[index] == 0] == float(white_background))
        assert full.dataset.sample(index).metadata["eval_background_color"] == (
            float(white_background),
        ) * 3
    from msbench.core.config import Config, finalize_config
    from msbench.core.builder import build_adapter
    render_cfg = Config(finalize_config(dict(
        dataset=dict(name="dtu", type="colmap", root=str(scene), dtu_eval_mode="full",
                     background="white" if white_background else "black"),
        trainer=dict(type="exmesh", white_background=white_background),
        adapter=dict(type="exmesh", checkpoint=str(tmp_path / "resumed/ckpt/final.pt")),
    )))
    adapter = build_adapter(render_cfg.adapter)
    rendered = adapter.render(full.dataset.sample(0).camera.to("cuda"))
    outside = rendered.alpha == 0
    assert outside.any() and torch.all(rendered.rgb[outside] == float(white_background))


def test_2dgs_initializer_holds_out_test_views_with_official_dtu_defaults(tmp_path, monkeypatch):
    import json
    import msbench.trainers.exmesh_preprocess as pp
    from msbench.core.config import load_config

    class Captured(Exception):
        pass

    def capture(source, command, output):
        raise Captured(command)

    monkeypatch.setattr(pp.subprocess, "check_output", lambda *a, **k: pp.TWO_DGS_REVISION)
    monkeypatch.setattr(pp, "stage_colmap", lambda *a, **k: {"views": []})
    monkeypatch.setattr(pp, "_record_effective_arguments", capture)
    cfg = load_config("configs/exmesh/dtu/scan24.yaml")
    cfg["preprocess"].update(source_repo=str(tmp_path), output=str(tmp_path / "out"))

    with pytest.raises(Captured) as captured:
        pp.prepare_exmesh_initializer(cfg, dataset_root=str(tmp_path))
    command = captured.value.args[0]
    assert "--eval" not in command
    assert command[command.index("--lambda_dist") + 1] == "1000.0"
    settings = json.loads((tmp_path / "out/settings.json").read_text())
    assert settings["holdout"] == 0 and settings["num_cluster"] == 1

    cfg["preprocess"].update(holdout=4, output=str(tmp_path / "out4"))
    with pytest.raises(ValueError, match="holdout 0 or 8"):
        pp.prepare_exmesh_initializer(cfg, dataset_root=str(tmp_path))
