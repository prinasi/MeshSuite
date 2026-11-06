#!/usr/bin/env python3
"""Bake a DiffSoup checkpoint into a standard engine deployment TriAsset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msbench.unity_assets import resolve_checkpoint  # noqa: E402
from msbench.unity_bake import bake_lattice_colors, write_baked_package  # noqa: E402

# The vendored trainer's near/far planes and held-out split.
TRAIN_Z_NEAR, Z_FAR, HOLDOUT = 0.01, 100.0, 8


def training_mvps(checkpoint: dict, dataset: Path, downscale: int) -> list[torch.Tensor]:
    """Rebuild the training views' MVPs with the vendored trainer's loader."""
    from msbench.vendor.diffsoup.examples import utils

    if str(checkpoint.get("dataset_type", "")).lower() != "colmap":
        raise ValueError(
            f"baking supports COLMAP scenes, got dataset_type={checkpoint.get('dataset_type')!r}"
        )
    scene = utils.load_mipnerf360_scene(str(dataset), split="train", holdout=HOLDOUT, downscale=downscale)
    height, width = int(checkpoint["H"]), int(checkpoint["W"])
    if (scene["H"], scene["W"]) != (height, width):
        raise ValueError(
            f"{dataset} at downscale {downscale} is {scene['W']}x{scene['H']}, "
            f"but the checkpoint was trained at {width}x{height}"
        )
    return [
        utils.mvp_from_K_Tcw(
            scene["K"].float(),
            frame["Tcw"].float(),
            (height, width),
            z_near=TRAIN_Z_NEAR,
            z_far=Z_FAR,
            flip_z=bool(checkpoint.get("flip_z", False)),
        )
        for frame in scene["frames"]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="final_params.pt or its run directory")
    parser.add_argument("--dataset", type=Path, required=True, help="COLMAP scene root")
    parser.add_argument("--downscale", type=int, default=4, help="trainer.downscale of the run")
    parser.add_argument("--output", type=Path, required=True, help="Output .triasset directory")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint("diffsoup", args.checkpoint)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    mvps = training_mvps(checkpoint, args.dataset, args.downscale)
    baked, stats = bake_lattice_colors(checkpoint, mvps)
    package = write_baked_package(checkpoint, checkpoint_path, baked, stats, args.output, overwrite=args.force)
    print(json.dumps({"status": "ok", "asset": str(package), **stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
