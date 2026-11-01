"""Tests for the T²-Splat integration (configs, upstream presets, runner, adapter)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from msbench.cli.train import inherit_dataset_defaults
from msbench.core.config import Config
from msbench.trainers.t2splat_native import geometry_argv, texture_settings

REPO = Path(__file__).resolve().parents[2]
PRESETS = json.loads((Path(__file__).with_name("data_t2splat_upstream_presets.json")).read_text())["scenes"]
# Options MeshSuite sets itself: total length comes from trainer.max_steps and the
# in-training test reports (logging only) are disabled.
RUNNER_OWNED = {"iterations", "final_only_monitoring"}


def _scene_configs():
    for path in sorted((REPO / "configs" / "t2splat").glob("*/*.yaml")):
        yield path.parts[-2], path.stem, path


def _trainer(cfg: Config) -> dict:
    return inherit_dataset_defaults(cfg.trainer.to_dict(), cfg.dataset.to_dict())


def _effective_options(argv: list[str]):
    """Resolve an upstream train.py argv exactly as upstream ``run_training`` does."""
    from msbench.vendor.t2splat.arguments import update_indoor
    from msbench.vendor.t2splat.train import build_parser

    parser, model_params, opt_params, _ = build_parser()
    args = parser.parse_args(argv)
    opt = opt_params.extract(args)
    if args.indoor:
        opt = update_indoor(opt)
    return model_params.extract(args), opt


def test_t2splat_covers_its_paper_datasets_only():
    datasets = {dataset for dataset, _, _ in _scene_configs()}
    # The paper evaluates Mip-NeRF 360, Tanks & Temples and DTU; no NeRF-Synthetic.
    assert datasets == {"mipnerf360", "tandt", "dtu"}
    assert {(d, s) for d, s, _ in _scene_configs()} == {(v["dataset"], s) for s, v in PRESETS.items()}


def test_t2splat_dataset_protocol_is_identical_to_other_methods():
    for dataset, scene, path in _scene_configs():
        cfg = Config.fromfile(str(path))
        other = Config.fromfile(str(REPO / "configs" / "mesh-splatting" / dataset / f"{scene}.yaml"))
        assert cfg.dataset.to_dict() == other.dataset.to_dict(), path
        # Without an explicit bg_color the adapter renders on the training background.
        white = cfg.trainer.to_dict().get("white_background", False)
        background = cfg.adapter.render_params.to_dict().get("bg_color", "white" if white else "black")
        assert background == other.adapter.render_params.bg_color, path
        assert white == other.trainer.to_dict().get("white_background", False), path
        assert cfg.output.dir == other.output.dir.replace("mesh-splatting", "t2splat"), path
        # No external priors for any method.
        assert cfg.trainer.disable_priors is True, path
        if dataset == "dtu":
            # DTU: every view trains, white foreground, 1/2 resolution, mesh metrics only.
            assert cfg.dataset.eval_split is False and cfg.dataset.resolution == 2
            assert cfg.dataset.dtu_eval_mode == "foreground"
            assert cfg.trainer.white_background is True
            assert cfg.mesh.split == "all"


def test_t2splat_method_bases_hold_no_dataset_protocol_fields():
    import yaml

    protocol = {"resolution", "image_dir", "images", "background", "white_background",
                "dtu_eval_mode", "eval_split", "eval_every", "llffhold"}
    for path in sorted((REPO / "configs" / "base").glob("t2splat*.yaml")):
        raw = yaml.safe_load(path.read_text())
        assert "_base_" not in raw, path
        assert "dataset" not in raw, path
        assert not protocol & set(raw.get("trainer", {})), path
        assert "bg_color" not in raw.get("adapter", {}).get("render_params", {}), path
        for section in ("render", "eval"):
            assert "split" not in raw.get(section, {}), path


@pytest.mark.parametrize("scene", sorted(PRESETS))
def test_t2splat_configs_reproduce_the_authors_repo_presets(scene, tmp_path):
    """MeshSuite configs resolve to upstream's effective per-scene train.py options."""
    from msbench.vendor.t2splat.train import build_parser

    preset = PRESETS[scene]
    cfg = Config.fromfile(str(REPO / "configs" / "t2splat" / preset["dataset"] / f"{scene}.yaml"))
    trainer = _trainer(cfg)
    texture = texture_settings(trainer)
    geometry_iterations = int(cfg.trainer.max_steps) - int(texture["iterations"])
    argv = geometry_argv(trainer, dataset_root=str(tmp_path), model_path=tmp_path / "ckpt",
                         geometry_iterations=geometry_iterations)
    model, opt = _effective_options(argv)
    _, defaults = _effective_options(["-s", str(tmp_path), "-m", str(tmp_path)])
    diff = {k: v for k, v in vars(opt).items() if vars(defaults).get(k) != v and k not in RUNNER_OWNED}
    expected = {k: v for k, v in preset["opt"].items() if k not in RUNNER_OWNED}
    assert diff == pytest.approx(expected), scene
    assert geometry_iterations == preset["opt"].get("iterations", 30000)
    # Images follow the shared dataset protocol, which matches upstream's folders.
    assert model.images == preset["images"]
    parser_args = build_parser()[0].parse_args(argv)
    assert parser_args.indoor is preset["indoor"]

    if preset["texture"] is None:
        assert texture["iterations"] == 0
    else:
        upstream = dict(preset["texture"])
        assert texture["iterations"] == upstream.pop("stage2_iterations")
        upstream.pop("test_interval")  # test-view logging only
        assert {k: texture[k] for k in upstream} == pytest.approx(upstream)
        assert texture["texture_adam_offload"] is False
    assert cfg.mesh.clean is preset["export_clean"]
    assert cfg.eval.mesh.geometry_mode == "mesh"
    assert cfg.adapter.checkpoint == f"{cfg.output.dir}/ckpt/point_cloud/iteration_{cfg.trainer.max_steps}"


def test_t2splat_geometry_argv_maps_the_dataset_protocol(tmp_path):
    cfg = Config.fromfile(str(REPO / "configs/t2splat/dtu/scan24.yaml"))
    argv = geometry_argv(_trainer(cfg), dataset_root=str(tmp_path), model_path=tmp_path / "ckpt",
                         geometry_iterations=20000)
    model, opt = _effective_options(argv)
    assert model.resolution == 2 and model.images == "images"
    assert model.eval is False  # all DTU views train
    assert model.white_background is True and model.dtu_eval_mode == "foreground"
    assert model.disable_priors is True
    assert opt.iterations == 20000 and opt.final_only_monitoring is True
    assert argv[argv.index("--test_iterations") + 1] == "-1"
    assert argv[argv.index("--save_iterations") + 1] == "20000"

    garden = Config.fromfile(str(REPO / "configs/t2splat/mipnerf360/garden.yaml"))
    model, _ = _effective_options(geometry_argv(
        _trainer(garden), dataset_root=str(tmp_path), model_path=tmp_path, geometry_iterations=30000))
    assert model.eval is True and model.white_background is False and model.resolution == 1
    assert model.images == "images_4" and model.data_device == "cpu"


def test_t2splat_runner_rejects_unknown_and_reserved_options(tmp_path):
    base = {"type": "t2splat", "images": "images"}
    kwargs = dict(dataset_root=str(tmp_path), model_path=tmp_path, geometry_iterations=10)
    with pytest.raises(ValueError, match="Unknown T²-Splat trainer option"):
        geometry_argv({**base, "lambda_nromals": 0.1}, **kwargs)
    with pytest.raises(ValueError, match="set by MeshSuite"):
        geometry_argv({**base, "iterations": 5}, **kwargs)
    with pytest.raises(ValueError, match="cannot be disabled"):
        geometry_argv({**base, "heartbeat_use_qem": False}, **kwargs)
    with pytest.raises(ValueError, match="Unknown T²-Splat trainer.texture keys"):
        texture_settings({"texture": {"res": 4}})


def test_t2splat_runner_skips_finished_runs_and_reuses_stage_one(tmp_path, monkeypatch):
    import msbench.trainers.t2splat_native as native

    def touch(iteration):
        path = tmp_path / "ckpt" / "point_cloud" / f"iteration_{iteration}"
        path.mkdir(parents=True, exist_ok=True)
        (path / "point_cloud_state_dict.pt").write_bytes(b"x")

    calls = []
    monkeypatch.setattr(native, "_run_texture_stage", lambda *a, **k: (calls.append("texture"), touch(35)))
    trainer = {"type": "t2splat", "texture": {"iterations": 5}}
    touch(30)
    summary = native.run_t2splat_native_config(
        trainer_cfg=trainer, dataset_root=str(tmp_path), output_dir=tmp_path, max_steps=35, quiet=True)
    # Stage I checkpoint exists: only Stage II runs.
    assert calls == ["texture"] and summary["stages"] == ["texture"] and summary["start_step"] == 30
    summary = native.run_t2splat_native_config(
        trainer_cfg=trainer, dataset_root=str(tmp_path), output_dir=tmp_path, max_steps=35, quiet=True)
    assert calls == ["texture"] and summary["trained_steps"] == 0
    with pytest.raises(ValueError, match="must exceed"):
        native.run_t2splat_native_config(
            trainer_cfg=trainer, dataset_root=str(tmp_path), output_dir=tmp_path, max_steps=5, quiet=True)


def test_t2splat_mesh_clean_matches_upstream_create_ply():
    from msbench.renderers.t2splat_adapter import clean_t2splat_mesh

    # One 200-face strip and one isolated triangle (< 1% of the largest component).
    n = 101
    top = np.stack([np.arange(n), np.zeros(n), np.zeros(n)], 1)
    bottom = top + [0, 1, 0]
    vertices = np.concatenate([top, bottom, [[50, 50, 0], [51, 50, 0], [50, 51, 0]]]).astype(np.float32)
    faces = [[i, i + 1, n + i] for i in range(n - 1)] + [[i + 1, n + i + 1, n + i] for i in range(n - 1)]
    faces = np.array(faces + [[2 * n, 2 * n + 1, 2 * n + 2]])
    colors = np.tile(np.arange(len(vertices))[:, None] % 255, (1, 3)).astype(np.uint8)
    mesh = clean_t2splat_mesh(vertices, faces, colors)
    assert len(mesh.faces) == 200 and len(mesh.vertices) == 2 * n
    assert np.array_equal(np.asarray(mesh.visual.vertex_colors)[:, :3], colors[: 2 * n])


def test_dtu_mesh_export_metadata_tracks_cleaning(tmp_path):
    from msbench.cli.eval import _has_current_t2splat_export_metadata

    mesh = tmp_path / "mesh.ply"
    meta = tmp_path / "mesh.ply.msbench.json"
    assert not _has_current_t2splat_export_metadata(mesh, {"clean": True})
    meta.write_text(json.dumps({"msbench_mesh_export": {"method": "t2splat", "version": 1, "clean": True}}))
    assert _has_current_t2splat_export_metadata(mesh, {"clean": True})
    assert not _has_current_t2splat_export_metadata(mesh, {"clean": False})


def test_t2splat_backend_and_adapter_are_registered():
    from msbench.core.registry import get_adapter
    from msbench.renderers.backends import canonical_backend_name

    assert canonical_backend_name("T2-Splat") == "t2splat"
    assert get_adapter("t2splat").__name__ == "T2SplatAdapter"


def _t2splat_cuda_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from msbench.vendor._cmod.t2splat_rasterization import _C
    except ImportError:
        return False
    return _C is not None


@pytest.mark.skipif(not _t2splat_cuda_available(), reason="T²-Splat requires its CUDA rasterizer")
def test_t2splat_adapter_renders_mip_textures_with_gradients(tmp_path):
    from msbench.core.cameras import CameraBatch
    from msbench.renderers.t2splat_adapter import T2SplatAdapter

    # A 2x2 quad grid in front of the camera, with residual 4x4 Mip-textures.
    xs, ys = np.meshgrid(np.linspace(-0.5, 0.5, 3), np.linspace(-0.5, 0.5, 3))
    vertices = torch.tensor(np.stack([xs.ravel(), ys.ravel(), np.zeros(9)], 1), dtype=torch.float32)
    faces = []
    for r in range(2):
        for c in range(2):
            a, b, d, e = r * 3 + c, r * 3 + c + 1, (r + 1) * 3 + c, (r + 1) * 3 + c + 1
            faces += [[a, b, d], [b, e, d]]
    faces = torch.tensor(faces, dtype=torch.int32)
    state = {
        "triangles_points": vertices, "_triangle_indices": faces,
        "vertex_weight": torch.full((9, 1), 8.0), "sigma": torch.tensor(np.log(1e-4)),
        "active_sh_degree": 0, "features_dc": torch.full((9, 1, 3), 0.5),
        "features_rest": torch.zeros(9, 15, 3), "importance_score": torch.ones(8),
        "image_size": torch.zeros(8), "pixel_count": torch.zeros(8),
        "face_textures": torch.zeros(8, 4, 4, 3), "face_tex_res": 4, "use_mip_texture": True,
    }
    ckpt = tmp_path / "ckpt" / "point_cloud" / "iteration_7"
    ckpt.mkdir(parents=True)
    torch.save(state, ckpt / "point_cloud_state_dict.pt")

    adapter = T2SplatAdapter()
    adapter.configure(bg_color="black", render_scaling=1)
    adapter.load_checkpoint(str(ckpt))
    view = torch.eye(4)[None]
    view[0, 2, 3] = 2.0
    k = torch.tensor([[[40.0, 0, 16], [0, 40, 12], [0, 0, 1]]])
    camera = CameraBatch(view, torch.linalg.inv(view), k, 32, 24)
    base = adapter.render(camera).rgb
    assert base.shape == (24, 32, 3) and torch.isfinite(base).all()
    assert base[12, 16].mean() > 0.5 and base[0, 0].abs().max() < 1e-3  # mesh vs background

    with torch.no_grad():
        adapter._model.face_textures += 0.2  # residual texture shifts the colour
    shifted = adapter.render(camera, mode="train").rgb
    assert torch.allclose(shifted[12, 16], base[12, 16] + 0.2, atol=0.02)
    shifted.mean().backward()
    grad = adapter._model.face_textures.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0

    stats = adapter.model_stats()
    assert stats["face_count"] == 8 and stats["backend_extra"]["face_tex_res"] == 4
    mesh_path = adapter.export_mesh(tmp_path / "mesh.ply", clean=True, dataset_path="ignored")
    meta = json.loads(mesh_path.with_suffix(".ply.msbench.json").read_text())["msbench_mesh_export"]
    assert meta["method"] == "t2splat" and meta["faces"] == 8 and meta["clean"] is True


def _t2splat_state(faces: int = 6, res: int = 8, *, mip: bool = True) -> dict:
    generator = torch.Generator().manual_seed(0)
    vertices = torch.rand(faces + 2, 3, generator=generator)
    tris = torch.tensor([[i, i + 1, i + 2] for i in range(faces)], dtype=torch.int32)
    return {
        "triangles_points": vertices, "_triangle_indices": tris,
        "vertex_weight": torch.randn(faces + 2, 1, generator=generator),
        "sigma": float(np.log(1e-4)), "active_sh_degree": 3,
        "features_dc": torch.randn(faces + 2, 1, 3, generator=generator),
        "features_rest": torch.randn(faces + 2, 15, 3, generator=generator),
        "face_textures": torch.randn(faces, res, res, 3, generator=generator) * 0.1,
        "face_tex_res": res, "use_mip_texture": mip,
    }


def _unpack_halves(words: np.ndarray, count: int) -> np.ndarray:
    halves = np.stack([words & 0xFFFF, words >> 16], axis=1).reshape(-1).astype(np.uint16)
    return halves.view(np.float16)[:count].astype(np.float32)


def test_t2splat_unity_mip_pyramid_matches_the_training_renderer():
    from msbench.unity_assets import t2splat_mip_pyramid
    from msbench.vendor.t2splat.scene.triangle_model import TriangleModel

    textures = torch.randn(5, 8, 8, 3)
    packed, resolutions, offsets = t2splat_mip_pyramid(textures)
    model = TriangleModel(3)
    model.face_textures = textures
    native, native_offsets, levels = model.build_mip_pyramid()
    assert resolutions == [8, 4, 2, 1] and levels == 4
    assert offsets == native_offsets.tolist() == [0, 64, 80, 84]
    assert torch.equal(packed.reshape(-1), native)


@pytest.mark.parametrize("mip", [True, False])
def test_t2splat_unity_export_packs_residual_mip_textures(tmp_path, mip):
    from msbench.unity_assets import export_triasset
    from tools.validate_unity_triasset_cpu import validate_triasset

    state = _t2splat_state(mip=mip)
    ckpt = tmp_path / "ckpt" / "point_cloud" / "iteration_35000"
    ckpt.mkdir(parents=True)
    torch.save(state, ckpt / "point_cloud_state_dict.pt")
    package = export_triasset("T2-Splat", tmp_path, tmp_path / "scene")
    manifest = json.loads(package.manifest_path.read_text())
    rendering = manifest["rendering"]
    assert manifest["method"] == "t2splat" and rendering["renderer"] == "t2splat"
    assert rendering["opacity_floor"] == 0.999 and rendering["terminal_solid_eligible"] is True
    assert rendering["standard_engine"]["supported"] is True
    assert rendering["native_render_scaling"] == 4
    levels = [8, 4, 2, 1] if mip else [8]
    assert rendering["face_mip_resolutions"] == levels
    texels = sum(r * r for r in levels)
    assert rendering["face_mip_texels_per_face"] == texels and rendering["face_mip_levels"] == len(levels)

    words = np.fromfile(package.path / "buffers" / "face_mip_textures.bin", dtype=np.uint32)
    assert words.size == (6 * texels * 3 + 1) // 2
    decoded = _unpack_halves(words, 6 * texels * 3).reshape(6, texels, 3)
    textures = state["face_textures"]
    np.testing.assert_allclose(decoded[:, :64], textures.reshape(6, 64, 3).numpy(), atol=1e-3)
    if mip:  # level 1 = 2x2 average of level 0
        level1 = torch.nn.functional.avg_pool2d(textures.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
        np.testing.assert_allclose(decoded[:, 64:80], level1.reshape(6, 16, 3).numpy(), atol=1e-3)
    opacity = np.fromfile(package.path / "buffers" / "triangle_opacity.bin", dtype=np.float32)
    weights = 0.999 + 0.001 / (1 + np.exp(-state["vertex_weight"].numpy().reshape(-1)))
    np.testing.assert_allclose(opacity, weights[state["_triangle_indices"].numpy()].min(1), rtol=1e-6)
    assert validate_triasset(package.path)["method"] == "t2splat"


def test_t2splat_unity_renderer_and_shader_are_wired():
    renderer = (REPO / "unity/Assets/MeshSuite/Scripts/MethodSpecificSplatRenderer.cs").read_text()
    shader = (REPO / "unity/Assets/MeshSuite/Shaders/T2SplatTerminalSolid.shader").read_text()
    meta = (REPO / "unity/Assets/MeshSuite/Shaders/T2SplatTerminalSolid.shader.meta").read_text()
    graphics = (REPO / "unity/ProjectSettings/GraphicsSettings.asset").read_text()
    assert 'method == "t2splat") mode = 3' in renderer and "MeshSuite/T2SplatTerminalSolid" in renderer
    for name in ("_FaceMip", "_HasFaceTexture", "_FaceTexRes", "_FaceMipLevels", "_FaceTexelsPerFace", "_LodPixelScale"):
        assert f'"{name}"' in renderer and name in shader
    guid = next(line.split()[1] for line in meta.splitlines() if line.startswith("guid:"))
    assert f"guid: {guid}, type: 3" in graphics  # always included in standalone players
    for path in ("Scripts/TriAssetRuntimeBatchBootstrap.cs", "Editor/TriAssetBatchRunner.cs"):
        assert '"t2splat"' in (REPO / "unity/Assets/MeshSuite" / path).read_text()
