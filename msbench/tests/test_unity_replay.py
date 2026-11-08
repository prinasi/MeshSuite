"""CPU tests for the Unity camera-replay scene of NeRF-Synthetic datasets."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from msbench.core.datasets import NerfSyntheticDataset
from msbench.unity_replay import REPLAY_IMAGE_DIR, replay_is_current, rotation_to_qvec, write_synthetic_replay
from msbench.vendor.triangle_splatting.scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
)

REPO = Path(__file__).resolve().parents[2]


def _make_scene(root: Path, count: int = 3) -> Path:
    (root / "test").mkdir(parents=True)
    frames = []
    for index in range(count):
        angle = 0.7 * index + 0.3
        c2w = np.eye(4)
        c2w[:3, :3] = np.array(
            [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        c2w[:3, 3] = [4.0 * math.sin(angle), -4.0 * math.cos(angle), 0.5 + index]
        frames.append({"file_path": f"./test/r_{index}", "transform_matrix": c2w.tolist()})
        rgba = np.zeros((8, 6, 4), dtype=np.uint8)
        rgba[..., 0] = 200
        rgba[..., 3] = 128
        rgba[0, 0, 3] = 0
        rgba[0, 1, 3] = 255
        Image.fromarray(rgba, "RGBA").save(root / "test" / f"r_{index}.png")
        Image.new("L", (6, 8)).save(root / "test" / f"r_{index}_depth_0000.png")
    (root / "transforms_test.json").write_text(json.dumps({"camera_angle_x": 0.69, "frames": frames}))
    return root


def test_rotation_to_qvec_round_trips_through_colmap():
    for seed in range(20):
        q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
        rotation = q * np.sign(np.linalg.det(q))
        qvec = rotation_to_qvec(rotation)
        assert qvec[0] >= 0
        np.testing.assert_allclose(qvec2rotmat(qvec), rotation, atol=1e-10)


def test_replay_cameras_match_the_native_loader(tmp_path: Path):
    scene = _make_scene(tmp_path / "chair")
    replay = write_synthetic_replay(scene, tmp_path / "replay/chair")

    cameras = read_intrinsics_binary(replay / "sparse/0/cameras.bin")
    images = {image.name: image for image in read_extrinsics_binary(replay / "sparse/0/images.bin").values()}
    assert len(cameras) == 1
    camera = cameras[1]
    assert (camera.model, camera.width, camera.height) == ("PINHOLE", 6, 8)

    native = NerfSyntheticDataset(scene, split="test", background="white", use_alpha=True)
    assert len(native) == len(images) == 3
    for index in range(len(native)):
        sample = native.sample(index)
        K = sample.camera.Ks[0].numpy()
        np.testing.assert_allclose(camera.params, [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], rtol=1e-6)
        image = images[f"{Path(sample.name).stem}.png"]
        w2c = sample.camera.viewmats[0].numpy()
        np.testing.assert_allclose(qvec2rotmat(image.qvec), w2c[:3, :3], atol=1e-5)
        np.testing.assert_allclose(image.tvec, w2c[:3, 3], atol=1e-5)


@pytest.mark.parametrize("background, empty", [("white", 255), ("black", 0)])
def test_ground_truth_is_composited_and_excludes_auxiliary_images(tmp_path: Path, background: str, empty: int):
    scene = _make_scene(tmp_path / "chair")
    replay = write_synthetic_replay(scene, tmp_path / "replay", background=background)

    assert sorted(path.name for path in (replay / REPLAY_IMAGE_DIR).iterdir()) == ["r_0.png", "r_1.png", "r_2.png"]
    with Image.open(replay / REPLAY_IMAGE_DIR / "r_0.png") as image:
        assert image.mode == "RGB"
        pixels = np.asarray(image)
    assert pixels[0, 0].tolist() == [empty, empty, empty]
    assert pixels[0, 1].tolist() == [200, 0, 0]
    alpha = 128 / 255
    expected = np.round(np.array([200 / 255, 0.0, 0.0]) * alpha * 255 + empty * (1 - alpha))
    np.testing.assert_allclose(pixels[1, 1], expected, atol=1)


def test_replay_is_reused_until_its_source_or_background_changes(tmp_path: Path):
    scene = _make_scene(tmp_path / "chair")
    replay = write_synthetic_replay(scene, tmp_path / "replay")
    marker = replay / "marker"
    marker.touch()
    write_synthetic_replay(scene, replay)
    assert marker.exists()

    assert not replay_is_current(scene, replay, "black")
    (scene / "transforms_test.json").write_text(
        json.dumps({**json.loads((scene / "transforms_test.json").read_text()), "camera_angle_x": 0.5})
    )
    assert not replay_is_current(scene, replay, "white")
    write_synthetic_replay(scene, replay)
    assert not marker.exists()
    assert replay_is_current(scene, replay, "white")


def _wrapper_dry_run(tmp_path: Path, target: str, scene_root: Path) -> str:
    dataset, scene = target.split("/")
    config = tmp_path / "configs" / "triangle-splatting" / dataset / f"{scene}.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        f"dataset:\n  root: {scene_root}\n  background: white\n"
        f"output:\n  dir: {tmp_path / 'out' / scene}\n"
    )
    return subprocess.run(
        [
            "bash", str(REPO / "single_unity_eval.sh"), "triangle-splatting", target, "0",
            "--config-root", str(tmp_path / "configs"), "--python", sys.executable,
            "--unity", str(tmp_path / "unity"), "--unity-project", str(tmp_path),
            "--skip-player-build", "--unity-player", str(tmp_path / "player"),
            "--skip-unity-patch", "--dry-run",
        ],
        cwd=REPO, capture_output=True, text=True, check=True,
        env={**os.environ, "UNITY_REPLAY_ROOT": str(tmp_path / "replay")},
    ).stdout


def test_eval_wrapper_replays_synthetic_scenes_on_every_test_view(tmp_path: Path):
    scene = _make_scene(tmp_path / "data" / "chair")
    out = _wrapper_dry_run(tmp_path, "nerf_synthetic/chair", scene)
    replay = tmp_path / "replay" / "nerf_synthetic" / "chair"

    assert f"tools/prepare_unity_replay.py --dataset {scene} --output {replay} --background white" in out
    assert f"--datasets-root {replay.parent}" in out
    assert out.count("--holdout 1") == 4  # One capture and three profile runs.
    assert "--holdout 8" not in out
    assert f"--ground-truth {replay / 'images'}" in out


def test_eval_wrapper_keeps_the_llff_holdout_on_colmap_scenes(tmp_path: Path):
    scene = tmp_path / "data" / "garden"
    (scene / "sparse" / "0").mkdir(parents=True)
    (scene / "sparse" / "0" / "cameras.bin").touch()
    (scene / "sparse" / "0" / "images.bin").touch()
    (scene / "images").mkdir()
    Image.new("RGB", (4, 4)).save(scene / "images" / "a.png")
    out = _wrapper_dry_run(tmp_path, "mipnerf360/garden", scene)

    assert "tools/prepare_unity_replay.py" not in out
    assert f"--datasets-root {scene.parent}" in out
    assert out.count("--holdout 8") == 4
    assert "--holdout 1 " not in out


def test_capture_reads_the_holdout_option():
    source = (REPO / "unity/Assets/MeshSuite/Scripts/ColmapBatchCapture.cs").read_text()
    assert 'Holdout = Mathf.Max(1, TriAssetRuntimeOptions.GetInt("-holdout", Holdout));' in source
