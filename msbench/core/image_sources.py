"""Shared, runtime-independent rules for prepared COLMAP image pyramids."""

from pathlib import Path
from hashlib import sha256
import json
import os
import re


COLMAP_TYPES = {"colmap", "mipnerf360", "tanks", "tanksandtemples", "tankstemple", "tandt"}
LEGACY_PREPROCESSING = "legacy_bilinear_floor"
LEGACY_PROTOCOL = "pil-bilinear-floor-rgb-v1"


def legacy_image_cache_dir(root: str | Path, factor: int) -> Path:
    root = Path(root).expanduser().resolve()
    base = Path(os.environ.get("MSBENCH_IMAGE_CACHE_DIR", ".cache/msbench/images")).expanduser().resolve()
    identity = sha256(str(root).encode()).hexdigest()[:16]
    return base / f"{root.name}-{identity}" / f"{LEGACY_PROTOCOL}-x{factor}"


def is_legacy_image_cache(image_dir: str | Path) -> bool:
    return re.fullmatch(LEGACY_PROTOCOL + r"-x[248]", Path(image_dir).name) is not None


def validate_legacy_image_cache(root: str | Path, directory: str | Path) -> dict:
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(
            f"Missing/invalid 2DTS preprocessing cache manifest: {manifest_path}. "
            "Run tools/prepare_2dts_images.py before training/evaluation."
        ) from exc
    if manifest.get("protocol") != LEGACY_PROTOCOL or manifest.get("source_root") != str(Path(root).resolve()):
        raise ValueError(f"Preprocessing cache protocol/source mismatch: {manifest_path}")
    if directory.name != f"{LEGACY_PROTOCOL}-x{manifest.get('factor')}":
        raise ValueError(f"Preprocessing cache factor mismatch: {manifest_path}")
    entries = manifest.get("images", [])
    if not entries:
        raise ValueError(f"Empty preprocessing cache: {manifest_path}")
    for entry in entries:
        source = Path(root) / "images" / entry["name"]
        stat = source.stat()
        if stat.st_size != entry["source_size"] or stat.st_mtime_ns != entry["source_mtime_ns"]:
            raise ValueError(f"Source image changed; regenerate the preprocessing cache: {source}")
        if not (directory / (entry["name"] + ".png")).is_file():
            raise FileNotFoundError(f"Cached preprocessing image missing: {entry['name']}.png")
    return manifest


def _standard_factor(resolution) -> int | None:
    """Return 2/4/8 for a standard downscale factor; None for widths, -1 or [W, H]."""
    if isinstance(resolution, bool) or not isinstance(resolution, (int, str)):
        return None
    try:
        factor = int(resolution)
    except ValueError:
        return None
    return factor if factor in {2, 4, 8} else None


def resolve_colmap_image_source(
    image_dir: str, resolution, *, root: str | Path | None = None,
    image_preprocessing: str = "stored",
) -> tuple[str, object]:
    """Resolve standard downscale factors to disk images, without resizing twice.

    Explicit custom folders and target-width/automatic resolution controls keep
    their existing meaning. Existence checks belong to dataset loading, so config
    resolution also works in the lightweight controller and without mounted data.
    """
    if image_preprocessing not in {"stored", LEGACY_PREPROCESSING}:
        raise ValueError(f"Unknown image preprocessing protocol: {image_preprocessing}")
    factor = _standard_factor(resolution)
    if image_preprocessing == LEGACY_PREPROCESSING and image_dir == "images" and factor is not None:
        if root is None:
            raise ValueError("Dataset root is required for the legacy preprocessing cache")
        return str(legacy_image_cache_dir(root, factor)), 1
    if image_dir == "images" and factor is not None:
        return f"images_{factor}", 1
    return image_dir, resolution


def is_prepared_image_dir(image_dir: str) -> bool:
    return is_legacy_image_cache(image_dir) or re.fullmatch(r"images_[1-9]\d*", Path(image_dir).name) is not None


def require_prepared_image_dir(root: str | Path, image_dir: str) -> None:
    path = Path(root).expanduser() / image_dir
    if is_legacy_image_cache(image_dir):
        validate_legacy_image_cache(root, path)
        return
    if is_prepared_image_dir(image_dir) and not path.is_dir():
        raise FileNotFoundError(
            f"Prepared COLMAP images not found: {path}. Generate {image_dir} "
            "before training/evaluation; dynamic downsampling fallback is disabled."
        )


def is_colmap_scene(root: str | Path) -> bool:
    root = Path(root).expanduser()
    return (root / "sparse").is_dir() and not (root / "cameras.npz").is_file()


def resolve_scene_image_source(root: str | Path, image_dir: str, resolution: int) -> tuple[str, int]:
    if is_colmap_scene(root):
        return resolve_colmap_image_source(image_dir, resolution)
    return image_dir, resolution


def unity_reference_image_path(root: str | Path, method: str, folder: str) -> Path:
    """Keep standalone MipNeRF360 Unity tools on the native image protocol."""
    root = Path(root)
    if method.replace("_", "-").lower() in {"2dts", "d2ts"} and folder in {"images_2", "images_4"}:
        directory = legacy_image_cache_dir(root, int(folder.rsplit("_", 1)[1]))
    else:
        directory = root / folder
    require_prepared_image_dir(root, str(directory))
    return directory
