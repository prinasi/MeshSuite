"""Stage train-only synthetic inputs for independent PGSR and DA3 preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


def stage_colmap(dataset: Path, output: Path):
    """Translate known Blender cameras, without SfM or test-image access."""
    from scipy.spatial.transform import Rotation
    from plyfile import PlyData

    meta_path = dataset / "transforms_train.json"
    meta = json.loads(meta_path.read_text())
    images, sparse = output / "images", output / "sparse"
    images.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(exist_ok=True)
    camera_lines, image_lines, records = [], [], []
    names = set()
    for index, frame in enumerate(meta["frames"], 1):
        relative = Path(frame["file_path"]).with_suffix(".png")
        name = relative.name
        if name in names:
            raise ValueError(f"Duplicate training image basename: {name}")
        names.add(name)
        with Image.open(dataset / relative) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255
        rgb = rgba[..., :3] * rgba[..., 3:] + 1 - rgba[..., 3:]
        Image.fromarray(np.round(rgb * 255).astype(np.uint8)).save(images / name)
        height, width = rgb.shape[:2]
        focal = width / (2 * np.tan(meta["camera_angle_x"] / 2))
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        quat = Rotation.from_matrix(w2c[:3, :3]).as_quat()[[3, 0, 1, 2]]
        camera_lines.append(
            f"{index} PINHOLE {width} {height} {focal} {focal} {width / 2} {height / 2}"
        )
        pose = " ".join(map(str, [*quat, *w2c[:3, 3]]))
        image_lines.append(f"{index} {pose} {index} {name}\n\n")
        records.append(
            dict(
                name=Path(name).stem,
                path=str((images / name).resolve()),
                extrinsics=w2c.tolist(),
                intrinsics=[[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
            )
        )
    (sparse / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    (sparse / "images.txt").write_text("".join(image_lines))
    # Standard NeRF-Synthetic points3d.ply is an initialization cloud, never GT.
    cloud_path = dataset / "points3d.ply"
    cloud = PlyData.read(cloud_path)["vertex"].data
    with (sparse / "points3D.txt").open("w") as stream:
        for i, row in enumerate(cloud, 1):
            stream.write(
                f"{i} {row['x']} {row['y']} {row['z']} {int(row['red'])} {int(row['green'])} {int(row['blue'])} 0\n"
            )
    manifest = dict(
        dataset=str(dataset.resolve()),
        split="train",
        background="white",
        train_count=len(records),
        test_images_used=False,
        records=records,
        transforms_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(),
        initialization_cloud=str(cloud_path.resolve()),
        initialization_cloud_is_ground_truth=False,
    )
    (output / "synthetic_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def predict_depth(manifest, output, model_path, batch_size, disable_autocast_cache=False):
    import torch
    from depth_anything_3.api import DepthAnything3

    output.mkdir(parents=True, exist_ok=True)
    records = manifest["records"]
    batch_size = batch_size or len(records)
    status = dict(
        model=str(model_path),
        batch_size=batch_size,
        process_res=504,
        split="train",
        train_count=len(records),
        test_images_used=False,
        background=manifest["background"],
        completed=[],
        status="running",
        autocast_weight_cache=not disable_autocast_cache,
    )
    status_path = output / "da3_status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    started = time.monotonic()
    try:
        model = DepthAnything3.from_pretrained(str(model_path)).to("cuda")
        for first in range(0, len(records), batch_size):
            batch = records[first : first + batch_size]
            # The API enables its own BF16/FP16 autocast in forward(). This outer
            # disabled context changes only its inherited weight-cache flag;
            # preprocessing stays FP32 and model/input dtypes stay unchanged.
            with torch.autocast("cuda", enabled=False, cache_enabled=not disable_autocast_cache):
                prediction = model.inference(
                    [r["path"] for r in batch],
                    extrinsics=np.array([r["extrinsics"] for r in batch]),
                    intrinsics=np.array([r["intrinsics"] for r in batch]),
                    process_res=504,
                )
            if len(prediction.depth) != len(batch):
                raise ValueError("DA3 returned the wrong number of depth maps")
            for record, depth in zip(batch, prediction.depth):
                if depth.ndim != 2 or not np.isfinite(depth).all() or not (depth > 0).any():
                    raise ValueError(f"Invalid DA3 depth for {record['name']}")
                np.savez_compressed(output / f"{record['name']}.npz", depth=np.round(depth, 6))
                status["completed"].append(record["name"])
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            print(f"DA3: {len(status['completed'])}/{len(records)}", flush=True)
        status["status"] = "complete"
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["duration_seconds"] = time.monotonic() - started
        status["peak_cuda_memory_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        status_path.write_text(json.dumps(status, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--disable-autocast-cache", action="store_true")
    parser.add_argument("--batch-size", type=int, default=0, help="0 = all training views together")
    args = parser.parse_args()
    if args.batch_size < 0 or (args.depth and not args.model):
        parser.error("Depth generation needs --model and a nonnegative --batch-size")
    input_dir = args.output / "colmap-input"
    if args.depth:
        manifest = json.loads((input_dir / "synthetic_manifest.json").read_text())
        predict_depth(
            manifest, args.output / "da3", args.model, args.batch_size, args.disable_autocast_cache
        )
    else:
        manifest = stage_colmap(args.dataset, input_dir)
        print(f"Staged {manifest['train_count']} train views; no test views")


if __name__ == "__main__":
    main()
