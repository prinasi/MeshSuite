"""Prepare the historical 2DTS image protocol without runtime downsampling.

Opt-in: only scene configs that set dataset.image_preprocessing: legacy_bilinear_floor
with image_dir: images and resolution 2/4/8 are prepared; the defaults use stored images_N.

Example: python -m tools.prepare_2dts_images --config path/to/scene.yaml
"""

import argparse
from pathlib import Path

from msbench.core.config import Config
from msbench.core.image_sources import COLMAP_TYPES, LEGACY_PREPROCESSING
from msbench.core.preprocessing import prepare_legacy_image_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Scene YAML or directory of scene YAMLs")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    paths = sorted(args.config.glob("*.yaml")) if args.config.is_dir() else [args.config]
    if not paths:
        parser.error("No scene configurations found")
    for path in paths:
        config = Config.fromfile(path)
        dataset = config.dataset
        if str(dataset.get("type", "auto")).lower() not in COLMAP_TYPES:
            continue
        if dataset.get("image_preprocessing") != LEGACY_PREPROCESSING:
            continue
        root = Path(dataset.root).expanduser()
        if dataset.get("scene") and root.name != Path(dataset.scene).name:
            root = root / dataset.scene
        factor = int(dataset.get("resolution", 1))
        if factor not in {2, 4, 8} or dataset.get("image_dir", "images") != "images":
            continue
        manifest = prepare_legacy_image_cache(root, factor, workers=args.workers)
        print(f"{path}: {len(manifest['images'])} images, factor {factor}, {manifest['protocol']}", flush=True)


if __name__ == "__main__":
    main()
