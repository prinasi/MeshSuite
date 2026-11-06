"""Mesh-oriented 2DTS: configs, the opacity threshold at export, and the wiring."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from msbench.core.config import load_config
from msbench.trainers.d2ts_native import _build_d2ts_native_config
from msbench.unity_assets import D2TS_SOLID_OPACITY_LOGIT, export_triasset
from tools.validate_unity_triasset_cpu import validate_triasset

REPO = Path(__file__).resolve().parents[2]
SCENES = {
    "mipnerf360": ("bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill"),
    "tandt": ("train", "truck"),
}
TARGETS = [(dataset, scene) for dataset, scenes in SCENES.items() for scene in scenes]


def _write_checkpoint(path: Path, opacity: list[float], *, gamma: float = 50.0) -> None:
    count = len(opacity)
    state = {
        "_vertex": torch.arange(count * 9, dtype=torch.float32).reshape(count, 3, 3),
        "_opacity": torch.logit(torch.tensor(opacity)).reshape(count, 1),
        "_f_dc": torch.arange(count * 3, dtype=torch.float32).reshape(count, 1, 3),
        "_f_rest": torch.zeros(count, 15, 3),
    }
    torch.save((state, {}, None, gamma), path)


@pytest.mark.parametrize("dataset,scene", TARGETS)
def test_opaque_scene_shares_the_volumetric_protocol_and_budget(dataset: str, scene: str):
    cfg = load_config(REPO / f"configs/2dts-opaque/{dataset}/{scene}.yaml")
    volumetric = load_config(REPO / f"configs/2dts/{dataset}/{scene}.yaml")

    assert cfg["dataset"] == volumetric["dataset"]
    assert cfg["d2ts"]["target_point_num"] == volumetric["d2ts"]["target_point_num"]
    assert cfg["d2ts"]["strategy"] == "opaque"
    assert volumetric["d2ts"]["strategy"] == "volumetric"
    assert cfg["trainer"]["type"] == cfg["adapter"]["type"] == "2dts"
    assert cfg["trainer"]["max_steps"] == 30_000

    render = cfg["adapter"]["render_params"]
    assert (render["ste_threshold"], render["sort_level"], render["gamma_rescale"]) == (0.3, 2, True)
    assert render["bg_color"] == volumetric["adapter"]["render_params"]["bg_color"]


@pytest.mark.parametrize("dataset,scene", TARGETS)
def test_opaque_outputs_never_touch_the_volumetric_run(dataset: str, scene: str):
    cfg = load_config(REPO / f"configs/2dts-opaque/{dataset}/{scene}.yaml")
    run = f"outputs/2dts-opaque/{dataset}/{scene}"

    assert cfg["output"]["dir"] == run
    assert cfg["adapter"]["checkpoint"] == f"{run}/ckpt"
    assert cfg["output"]["metrics_file"] == f"{run}/metrics.json"
    assert cfg["render"]["video"]["output_dir"] == f"{run}/video"
    assert f"outputs/2dts/{dataset}" not in yaml.safe_dump(cfg)


def test_primitive_cap_keeps_the_variant_in_the_ablation_path(tmp_path: Path):
    config = tmp_path / "capped.yaml"
    config.write_text(
        f"_base_: {REPO}/configs/2dts-opaque/tandt/truck.yaml\ntrainer:\n  max_primitives: 8000\n"
    )
    cfg = load_config(config)
    assert cfg["output"]["dir"] == "outputs/abl/2dts-opaque-8000/tandt/truck"
    assert cfg["d2ts"]["target_point_num"] == 8000


@pytest.mark.parametrize("dataset", SCENES)
def test_native_presets_follow_the_paper_schedule_and_upstream_recipe(dataset: str):
    native = yaml.safe_load((REPO / f"configs/2dts/native/{dataset}-opaque.yaml").read_text())
    model, update, trainer = native["model"], native["model"]["model_update"], native["trainer"]

    # Shared dataset protocol, not upstream's train-on-everything setting.
    assert native["dataset"]["hold_test_set"] is True
    assert native["dataset"]["hold_interval"] == 8
    # The paper's mesh setting (arXiv v3 Sec. 4.3); learning-rate decay ends at the anneal.
    assert update["gamma_schedule"] == {"start_iter": 20_000, "end_iter": 30_000, "gamma_init": 1.0, "gamma_final": 50.0}
    assert trainer["iterations"] == 30_000
    assert model["optimizer"]["vertex"]["max_steps"] == 20_000
    assert model["max_sh_degree"] == 3
    # Upstream config/TanksAndTemples_VanillaTS.yaml.
    assert (model["sort_level"], model["ste_threshold"], model["gamma_rescale"]) == (2, 0.3, True)
    assert model["sampling"]["init_opacity"] == 0.5
    assert (model["optimizer"]["vertex"]["v_init"], model["optimizer"]["vertex"]["v_final"]) == (0.0012, 0.00002)
    assert model["optimizer"]["opacity"]["v_final"] == 0.001
    assert model["optimizer"]["f_rest"]["v_init"] == 0.0002
    assert update["opacity_reset"]["reset_value"] == 0.29
    assert update["densification"]["split_scale_threshold"] == 0.01
    assert update["contribution_pruning"]["interval_iter"] == -1
    assert update["contribution_pruning"]["contrib_sum_threshold"] == 2.0
    assert "opacity_pruning" not in update
    assert trainer["smoothness_loss"]["w_normal"] == 0.05
    assert trainer["geometry_loss"]["w_geometry"] == 0.05
    assert trainer["distortion_loss"]["w_distortion"] == 0.01
    assert trainer["distortion_loss"]["start_iter"] == 6_000
    # MeshSuite scores no geometry on these datasets, and the zero-weight
    # consistency loss would only load neighbour views.
    assert trainer["save_pcd_iterations"] is None
    assert "neighbor_cams_args" not in native["dataset"]
    assert "consistency_loss" not in trainer

    expected_radii = {"tandt": 0.5, "mipnerf360": -1}[dataset]
    assert update["scale_pruning"]["radii_threshold"] == expected_radii


@pytest.mark.parametrize("dataset,scene", [("mipnerf360", "bicycle"), ("tandt", "truck")])
def test_resolved_native_config_solidifies_over_the_last_third(dataset: str, scene: str, tmp_path: Path, monkeypatch):
    monkeypatch.chdir(REPO)
    cfg = load_config(REPO / f"configs/2dts-opaque/{dataset}/{scene}.yaml")
    native = _build_d2ts_native_config(
        cfg=cfg,
        dataset_root=f"data/{dataset}/{scene}",
        output_dir=tmp_path / scene,
        max_steps=cfg["trainer"]["max_steps"],
    )

    assert native.trainer.iterations == 30_000
    assert native.model.ste_threshold == 0.3
    assert native.model.sort_level == 2
    assert native.model.model_update.gamma_schedule.start_iter == 20_000
    assert native.model.model_update.gamma_schedule.end_iter == 30_000
    assert native.model.model_update.densification.target_point_num == cfg["d2ts"]["target_point_num"]
    assert native.trainer.save_pcd_iterations is None
    assert native.trainer.save_mesh_iterations == [30_000]
    assert native.trainer.checkpoint_iterations == [30_000]
    assert native.trainer.eval_interval_iter == 0


def test_export_keeps_only_the_triangles_the_native_renderer_draws(tmp_path: Path):
    checkpoint = tmp_path / "30000.ckpt"
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


def _export_dry_run(method: str, tmp_path: Path) -> str:
    """Resolve the export command for truck against a checkpoint that exists."""
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    config = tmp_path / "configs" / method / "tandt" / "truck.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        f"_base_: {REPO}/configs/{method}/tandt/truck.yaml\n"
        f"adapter:\n  checkpoint: {checkpoint}\n"
        f"output:\n  dir: {tmp_path}/run\n"
    )
    result = subprocess.run(
        ["bash", "single_export_unity.sh", method, "tandt/truck", "cpu", "--dry-run",
         "--config-root", str(tmp_path / "configs"), "--python", sys.executable],
        cwd=REPO, check=True, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    return result.stdout + result.stderr


def test_export_wrapper_passes_the_threshold_and_the_engine_method(tmp_path: Path):
    output = _export_dry_run("2dts-opaque", tmp_path)
    assert "--method 2dts " in output
    assert "--d2ts-ste-threshold 0.3" in output
    assert f"{tmp_path}/run/unity_native/2dts" in output


def test_export_wrapper_leaves_the_volumetric_run_alone(tmp_path: Path):
    output = _export_dry_run("2dts", tmp_path)
    assert "--method 2dts " in output
    assert "--d2ts-ste-threshold" not in output


def _read(path: Path, dtype: str):
    import numpy as np

    return np.fromfile(path, dtype=dtype)
