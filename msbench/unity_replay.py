"""Derived camera-replay scenes for Unity deployment on non-COLMAP datasets.

Unity replays cameras from COLMAP binaries and scores against a flat image
directory. A NeRF-Synthetic scene has neither, and its test PNGs carry alpha that
the deployment evaluator would discard, so this module writes a separate scene
holding the official test views as PINHOLE COLMAP binaries plus the ground truth
composited on the dataset's background. The dataset itself is never modified.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from msbench.core.datasets import NerfSyntheticDataset

REPLAY_REVISION = 1
REPLAY_IMAGE_DIR = "images"
# Every replayed view is an official test view, so nothing may be held back.
REPLAY_HOLDOUT = 1
_COLMAP_PINHOLE = 1
_RECORD = "replay.json"


def rotation_to_qvec(rotation: np.ndarray) -> np.ndarray:
    """COLMAP's rotation-to-quaternion conversion, as ``(w, x, y, z)`` with ``w >= 0``."""
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = np.asarray(rotation, dtype=np.float64).flat
    k = np.array(
        [
            [rxx - ryy - rzz, 0.0, 0.0, 0.0],
            [ryx + rxy, ryy - rxx - rzz, 0.0, 0.0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0.0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ]
    ) / 3.0
    values, vectors = np.linalg.eigh(k)
    qvec = vectors[[3, 0, 1, 2], np.argmax(values)]
    return -qvec if qvec[0] < 0 else qvec


def _write_cameras(path: Path, width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<iiQQ", 1, _COLMAP_PINHOLE, width, height))
        handle.write(struct.pack("<4d", fx, fy, cx, cy))


def _write_images(path: Path, views: list[tuple[str, np.ndarray]]) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(views)))
        for image_id, (name, w2c) in enumerate(views, start=1):
            handle.write(struct.pack("<i", image_id))
            handle.write(struct.pack("<4d", *rotation_to_qvec(w2c[:3, :3])))
            handle.write(struct.pack("<3d", *np.asarray(w2c[:3, 3], dtype=np.float64)))
            handle.write(struct.pack("<i", 1))
            handle.write(name.encode("utf-8") + b"\0")
            handle.write(struct.pack("<Q", 0))


def _record(scene_root: Path, background: str) -> dict[str, object]:
    source = (scene_root / "transforms_test.json").read_bytes()
    return {
        "revision": REPLAY_REVISION,
        "background": background,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "views": len(json.loads(source)["frames"]),
    }


def replay_is_current(scene_root: Path, output: Path, background: str) -> bool:
    record_path = output / _RECORD
    if not record_path.is_file() or json.loads(record_path.read_text()) != _record(scene_root, background):
        return False
    views = json.loads(record_path.read_text())["views"]
    return (
        (output / "sparse/0/cameras.bin").is_file()
        and (output / "sparse/0/images.bin").is_file()
        and sum(1 for _ in (output / REPLAY_IMAGE_DIR).glob("*.png")) == views
    )


def write_synthetic_replay(
    scene_root: str | Path,
    output: str | Path,
    *,
    background: str = "white",
    force: bool = False,
) -> Path:
    """Write the replay scene for one NeRF-Synthetic scene and return its path.

    Poses, intrinsics and ground truth come from the loader native rendering and
    scoring use, so Unity sees the same views. An up-to-date scene is reused.
    """
    scene_root, output = Path(scene_root), Path(output)
    if not force and replay_is_current(scene_root, output, background):
        return output

    dataset = NerfSyntheticDataset(scene_root, split="test", background=background, use_alpha=True)
    staging = output.with_name(f"{output.name}.staging")
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "sparse/0").mkdir(parents=True)
    (staging / REPLAY_IMAGE_DIR).mkdir()

    views: list[tuple[str, np.ndarray]] = []
    intrinsics: set[tuple[int, int, float, float, float, float]] = set()
    for index in range(len(dataset)):
        sample = dataset.sample(index)
        camera = sample.camera
        K = camera.Ks[0].cpu().numpy().astype(np.float64)
        intrinsics.add((int(camera.width), int(camera.height), K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
        name = f"{Path(sample.name).stem}.png"
        views.append((name, camera.viewmats[0].cpu().numpy().astype(np.float64)))
        pixels = (sample.image[..., :3].cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
        Image.fromarray(pixels).save(staging / REPLAY_IMAGE_DIR / name)
    if len(intrinsics) != 1:
        raise ValueError(f"{scene_root} test views do not share one camera: {sorted(intrinsics)}")

    _write_cameras(staging / "sparse/0/cameras.bin", *intrinsics.pop())
    _write_images(staging / "sparse/0/images.bin", views)
    (staging / _RECORD).write_text(json.dumps(_record(scene_root, background), indent=2) + "\n")
    shutil.rmtree(output, ignore_errors=True)
    staging.rename(output)
    return output
