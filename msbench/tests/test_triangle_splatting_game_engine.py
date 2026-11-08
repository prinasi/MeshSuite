"""Triangle Splatting trained with the official train_game_engine.py recipe."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from msbench.cli.train import _triangle_splatting_native_argv
from msbench.core.config import load_config
from msbench.vendor.triangle_splatting.train import (
    build_parser,
    final_dead_mask,
    game_engine_opacity_dead,
)

REPO = Path(__file__).resolve().parents[2]
VARIANT = "triangle-splatting-game-engine"


def _triangles(opacity, sigma, *, importance=None, area=None):
    count = len(opacity)
    opacity = torch.tensor(opacity, dtype=torch.float32)[:, None]
    return SimpleNamespace(
        get_opacity=opacity,
        _opacity=opacity,
        get_sigma=torch.tensor(sigma, dtype=torch.float32)[:, None],
        importance_score=torch.tensor(importance or [1.0] * count)[:, None],
        triangle_area=torch.tensor(area or [10.0] * count)[:, None],
    )


OPT = SimpleNamespace(opacity_dead=0.014, importance_threshold=0.022)


def test_opacity_threshold_follows_train_game_engine():
    assert game_engine_opacity_dead(25_500) == 0.1
    assert game_engine_opacity_dead(29_000) == 0.1
    assert game_engine_opacity_dead(29_500) == 0.7


def test_default_pruning_is_unchanged():
    triangles = _triangles([0.01, 0.05, 0.5, 0.9], [0.5, 0.5, 1.5, 0.5], importance=[1, 1, 1, 0.01])
    mask = final_dead_mask(triangles, OPT, iteration=29_500, number_of_views=100, new_round=False)
    assert mask.tolist() == [True, False, False, True]


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [
        (26_000, [True, False, False, False]),  # opacity <= 0.1 only
        (28_500, [True, False, True, False]),  # plus sigma >= 1
        (29_500, [True, True, True, False]),  # opacity <= 0.7 and sigma >= 1
    ],
)
def test_game_engine_pruning_drops_transparent_and_soft_triangles(iteration, expected):
    triangles = _triangles([0.05, 0.5, 0.9, 0.9], [0.5, 0.5, 1.5, 0.5])
    mask = final_dead_mask(
        triangles, OPT, iteration=iteration, number_of_views=100, new_round=False, game_engine=True
    )
    assert mask.tolist() == expected


def test_game_engine_pruning_keeps_the_most_opaque_when_all_would_go():
    triangles = _triangles([0.2, 0.3, 0.4], [1.5, 1.5, 1.5])
    mask = final_dead_mask(
        triangles, OPT, iteration=29_500, number_of_views=100, new_round=False, game_engine=True
    )
    assert mask.tolist() == [True, True, False]


def test_trainer_flag_reaches_the_native_loop():
    parser, *_ = build_parser()
    assert parser.parse_args(["-s", "data", "--game_engine"]).game_engine
    assert not parser.parse_args(["-s", "data"]).game_engine

    def argv(trainer_cfg):
        return _triangle_splatting_native_argv(
            trainer_cfg=trainer_cfg, dataset_root="data/scene", output_dir=Path("/nonexistent"),
            max_steps=30_000, quiet=True,
        )

    assert "--game_engine" in argv({"game_engine": True})
    assert "--game_engine" not in argv({})


def _flat(mapping, prefix=""):
    out = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            out.update(_flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


BASE_SCENES = sorted(
    path.relative_to(REPO / "configs" / "triangle-splatting")
    for dataset in ("mipnerf360", "tandt", "nerf_synthetic")
    for path in (REPO / "configs" / "triangle-splatting" / dataset).glob("*.yaml")
)


def test_every_triangle_splatting_scene_has_a_game_engine_config():
    variant = sorted(path.relative_to(REPO / "configs" / VARIANT) for path in (REPO / "configs" / VARIANT).rglob("*.yaml"))
    assert variant == BASE_SCENES


@pytest.mark.parametrize("scene", BASE_SCENES, ids=str)
def test_game_engine_configs_differ_from_triangle_splatting_only_by_the_recipe(scene):
    base = _flat(load_config(REPO / "configs" / "triangle-splatting" / scene))
    variant = _flat(load_config(REPO / "configs" / VARIANT / scene))

    assert variant.pop("trainer.game_engine") is True
    assert variant.pop("output.method") == VARIANT
    assert variant.pop("experiment.name") == base.pop("experiment.name").replace(
        "triangle_splatting_", "triangle_splatting_game_engine_", 1
    )
    assert variant["output.dir"] == f"outputs/{VARIANT}/{scene.parent}/{scene.stem}"
    assert set(variant) == set(base)
    for key, value in base.items():
        if isinstance(value, str) and "outputs/triangle-splatting/" in value:
            value = value.replace("outputs/triangle-splatting/", f"outputs/{VARIANT}/")
        assert variant[key] == value, key


def _single_train(*args):
    return subprocess.run(
        ["bash", str(REPO / "single_train.sh"), *args, "--python", sys.executable],
        cwd=REPO, capture_output=True, text=True,
    )


def test_single_train_flag_selects_the_game_engine_configs(tmp_path: Path):
    result = _single_train("triangle-splatting", "mipnerf360/bicycle", "0", "--train_game_engine", "--config_root", str(tmp_path))
    assert result.returncode == 1
    assert f"Config missing: {tmp_path}/{VARIANT}/mipnerf360/bicycle.yaml" in result.stderr


def test_single_train_flag_rejects_other_methods():
    result = _single_train("2dts", "mipnerf360/bicycle", "0", "--train_game_engine")
    assert result.returncode == 1
    assert "--train_game_engine applies only to triangle-splatting" in result.stderr


@pytest.mark.parametrize("script", ["single_export_unity.sh", "single_unity_eval.sh", "single_unity_video.sh"])
def test_unity_scripts_map_every_variant_to_its_base_method(script):
    # A second engine_method() definition silently replaces the first in bash.
    assert (REPO / script).read_text().count("\nengine_method() {") == 1
    probe = (
        f'eval "$(sed -n "/^engine_method() {{/,/^}}/p" {script})"; '
        f"for m in 2dts-opaque {VARIANT} 2dts triangle-splatting; do engine_method $m; done"
    )
    out = subprocess.run(["bash", "-c", probe], cwd=REPO, capture_output=True, text=True, check=True).stdout
    assert out.split() == ["2dts", "triangle-splatting", "2dts", "triangle-splatting"]


def test_unity_eval_deploys_the_variant_as_triangle_splatting(tmp_path: Path):
    scene = tmp_path / "data" / "garden"
    (scene / "sparse" / "0").mkdir(parents=True)
    for name in ("cameras.bin", "images.bin"):
        (scene / "sparse" / "0" / name).touch()
    (scene / "images").mkdir()
    Image.new("RGB", (4, 4)).save(scene / "images" / "a.png")
    out_dir = tmp_path / "out" / VARIANT / "mipnerf360" / "garden"
    config = tmp_path / "configs" / VARIANT / "mipnerf360" / "garden.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(f"dataset:\n  root: {scene}\noutput:\n  dir: {out_dir}\n")
    out = subprocess.run(
        [
            "bash", str(REPO / "single_unity_eval.sh"), VARIANT, "mipnerf360/garden", "0",
            "--config-root", str(tmp_path / "configs"), "--python", sys.executable,
            "--unity", str(tmp_path / "unity"), "--unity-project", str(tmp_path),
            "--skip-player-build", "--unity-player", str(tmp_path / "player"),
            "--skip-unity-patch", "--dry-run",
        ],
        cwd=REPO, capture_output=True, text=True, check=True, env=dict(os.environ),
    ).stdout

    assert f"single_export_unity.sh {VARIANT} mipnerf360/garden" in out
    assert out.count("--method triangle-splatting ") == 4  # capture and three profile runs
    assert f"--outputs-root {out_dir.parent}" in out
    assert f"--method {VARIANT} --condition custom-engine" in out
    assert f"outputs/unity_reports/{VARIANT}/unity_custom_engine" in out
