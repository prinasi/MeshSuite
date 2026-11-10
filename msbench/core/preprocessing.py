"""Offline, lossless reproduction of the historical 2DTS COLMAP inputs."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image, __version__ as pillow_version

from msbench.core.image_sources import LEGACY_PROTOCOL, legacy_image_cache_dir, validate_legacy_image_cache


def legacy_bilinear_pixels(image: Image.Image, factor: int) -> np.ndarray:
    """Match native RGB channel loading, bilinear interpolation and floor size."""
    size = (image.width // factor, image.height // factor)
    if min(size) < 1:
        raise ValueError(f"Image {image.size} is too small for downscale factor {factor}")
    channels = image.split()
    channels = channels[:3] if len(channels) >= 3 else (channels[0],) * 3
    if image.size != size:
        channels = tuple(channel.resize(size, Image.Resampling.BILINEAR) for channel in channels)
    return np.stack([np.asarray(channel, dtype=np.uint8) for channel in channels], axis=-1)


def prepare_legacy_image_cache(root: str | Path, factor: int, *, workers: int = 4) -> dict:
    from msbench.vendor.d2ts.diff_recon.datasets.colmap_loader import read_extrinsics_binary, read_extrinsics_text

    root = Path(root).expanduser().resolve()
    if factor not in {2, 4, 8}:
        raise ValueError("Legacy image cache supports factors 2, 4 and 8")
    destination = legacy_image_cache_dir(root, factor)
    if destination.exists():
        return validate_legacy_image_cache(root, destination)
    sparse = root / "sparse/0"
    if not sparse.is_dir():
        sparse = root / "sparse"
    images = (read_extrinsics_binary(str(sparse / "images.bin")) if (sparse / "images.bin").is_file()
              else read_extrinsics_text(str(sparse / "images.txt")))
    names = sorted(image.name for image in images.values())
    if not names or len(set(names)) != len(names):
        raise ValueError("COLMAP images must have unique, nonempty names")
    for name in names:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Invalid COLMAP image name: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        def prepare(name):
            source = root / "images" / name
            before = source.stat()
            data = source.read_bytes()
            with Image.open(BytesIO(data)) as image:
                original_size = list(image.size)
                pixels = legacy_bilinear_pixels(image, factor)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Source changed during preprocessing: {source}")
            output = temporary / (name + ".png")
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(output, format="PNG", compress_level=1)
            # Verify the saved lossless artifact rather than assuming encoder fidelity.
            with Image.open(output) as image:
                if not np.array_equal(np.asarray(image), pixels):
                    raise ValueError(f"Lossless preprocessing verification failed: {output}")
            return {"name": name, "source_size": before.st_size, "source_mtime_ns": before.st_mtime_ns,
                    "source_sha256": sha256(data).hexdigest(), "original_size": original_size,
                    "size": [pixels.shape[1], pixels.shape[0]],
                    "pixels_sha256": sha256(pixels.tobytes()).hexdigest()}

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            entries = list(executor.map(prepare, names))
        manifest = {"schema_version": 1, "protocol": LEGACY_PROTOCOL, "factor": factor,
                    "pillow_version": pillow_version, "source_root": str(root), "images": entries}
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.rename(destination)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
