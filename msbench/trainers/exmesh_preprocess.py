"""Independent, versioned 2DGS initialization for the ExMesh paper protocol."""

from __future__ import annotations

import hashlib
import importlib.util
import argparse
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

TWO_DGS_URL = "https://github.com/hbb1/2d-gaussian-splatting.git"
TWO_DGS_REVISION = "f3e3b9fa67bbd1c75e05167ff37391d8dab2a678"


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _record_effective_arguments(source, command, output):
    """Capture defaults from the pinned source instead of retyping them."""
    spec = importlib.util.spec_from_file_location(
        "_msbench_2dgs_arguments", source / "arguments/__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = argparse.ArgumentParser()
    module.ModelParams(parser)
    module.OptimizationParams(parser)
    module.PipelineParams(parser)
    arguments, _ = parser.parse_known_args(command[1:])
    files = (
        "train.py",
        "render.py",
        "arguments/__init__.py",
        "utils/mesh_utils.py",
        "utils/camera_utils.py",
        "gaussian_renderer/__init__.py",
    )
    result = dict(
        arguments=vars(arguments), source_files_sha256={f: _sha256(source / f) for f in files}
    )
    (output / "effective_arguments.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def stage_colmap(dataset, output, resolution, *, background=None):
    """Resize/compose RGB, retain alpha for TSDF, and scale camera intrinsics."""
    from msbench.core.datasets import _resolve_resolution
    from msbench.vendor.triangle_splatting.scene.colmap_loader import (
        read_extrinsics_binary,
        read_extrinsics_text,
        read_intrinsics_binary,
        read_intrinsics_text,
    )

    if background not in {None, "black", "white"}:
        raise ValueError(f"Unsupported 2DGS input background: {background}")
    dataset, output = Path(dataset), Path(output)
    sparse = dataset / "sparse/0"
    if not sparse.is_dir():
        sparse = dataset / "sparse"
    binary = (sparse / "cameras.bin").is_file() and (sparse / "images.bin").is_file()
    cameras = (
        read_intrinsics_binary(sparse / "cameras.bin")
        if binary
        else read_intrinsics_text(sparse / "cameras.txt")
    )
    images = (
        read_extrinsics_binary(sparse / "images.bin")
        if binary
        else read_extrinsics_text(sparse / "images.txt")
    )
    target_sparse = output / "sparse/0"
    target_sparse.mkdir(parents=True, exist_ok=True)
    records, camera_lines, image_lines = [], [], []
    for index, image in enumerate(sorted(images.values(), key=lambda item: item.name), 1):
        camera = cameras[image.camera_id]
        source_image = dataset / "images" / image.name
        with Image.open(source_image) as source:
            width, height = _resolve_resolution(*source.size, resolution)
            channels = [
                channel.resize((width, height)) for channel in source.convert("RGBA").split()
            ]
            if background is not None:
                # Official 2DGS's COLMAP loader consumes RGB without alpha compositing.
                # Bake the target background once, but keep alpha for TSDF masking.
                rgb = np.stack([np.asarray(c) for c in channels[:3]], axis=-1).astype(np.float32)
                alpha = np.asarray(channels[3], dtype=np.float32)[..., None] / 255.0
                rgb = rgb * alpha + (255.0 if background == "white" else 0.0) * (1 - alpha)
                channels[:3] = Image.fromarray(np.rint(rgb).astype(np.uint8)).split()
            target = output / "images" / image.name
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.merge("RGBA", channels).save(target)
        if camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        elif camera.model == "SIMPLE_PINHOLE":
            focal, cx, cy = camera.params
            fx = fy = focal
        else:
            raise ValueError(
                f"2DGS initialization needs undistorted pinhole cameras: {camera.model}"
            )
        sx, sy = width / camera.width, height / camera.height
        intrinsics = [fx * sx, fy * sy, cx * sx, cy * sy]
        camera_lines.append(f"{index} PINHOLE {width} {height} " + " ".join(map(str, intrinsics)))
        pose = " ".join(map(str, [*image.qvec, *image.tvec]))
        image_lines.append(f"{image.id} {pose} {index} {image.name}\n\n")
        records.append(
            dict(
                name=image.name,
                width=width,
                height=height,
                input_sha256=_sha256(source_image),
                intrinsics=intrinsics,
            )
        )
    (target_sparse / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    (target_sparse / "images.txt").write_text("".join(image_lines))
    for name in ("points3D.bin", "points3D.txt", "points3D.ply"):
        source, target = sparse / name, target_sparse / name
        if source.is_file() and not target.exists():
            target.symlink_to(source.resolve())
    if not records:
        raise ValueError("No training images for 2DGS initialization")
    manifest = dict(
        views=records,
        split="all",
        resolution=resolution,
        background=background,
        source=str(dataset.resolve()),
        source_intrinsics_sha256=_sha256(sparse / ("cameras.bin" if binary else "cameras.txt")),
        source_poses_sha256=_sha256(sparse / ("images.bin" if binary else "images.txt")),
        source_cloud_sha256={
            name: _sha256(sparse / name)
            for name in ("points3D.bin", "points3D.txt", "points3D.ply")
            if (sparse / name).is_file()
        },
    )
    (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_2dgs(config, *, dataset_root):
    """Run/cache the separate initializer before the native ExMesh training loop."""
    from msbench.core.config import Config

    config = Config(config).to_dict()
    options = config["preprocess"]
    if options.get("type") == "exmesh_pgsr":
        return prepare_pgsr(config, dataset_root=dataset_root)
    if options["type"] != "exmesh_2dgs":
        raise ValueError(f"Unsupported preprocessing type: {options['type']}")
    background = "white" if config["trainer"].get("white_background", False) else "black"
    if config["dataset"].get("background", background) != background:
        raise ValueError("ExMesh initialization, training and evaluation backgrounds must agree")
    if importlib.util.find_spec("diff_surfel_rasterization") is None:
        raise ImportError(
            "2DGS preprocessing requires the official diff_surfel_rasterization extension; "
            "see 'ExMesh DTU protocols' in README.md"
        )
    output = Path(options["output"]).resolve()
    source = Path(options["source_repo"]).resolve()
    revision = options.get("source_revision", TWO_DGS_REVISION)
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", TWO_DGS_URL, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", revision], check=True)
        subprocess.run(
            ["git", "-C", str(source), "submodule", "update", "--init", "--recursive"], check=True
        )
    actual_revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_revision != revision:
        raise ValueError(f"2DGS source revision mismatch: {actual_revision} != {revision}")
    settings = dict(
        source_revision=revision,
        dataset=str(Path(dataset_root).resolve()),
        resolution=config["dataset"]["resolution"],
        iterations=7000,
        mesh_resolution=256,
        lambda_dist=options["lambda_dist"],
        depth_ratio=1.0,
        num_cluster=options["num_cluster"],
        background=background,
        rgba_compositing="baked_rgb_preserved_alpha_v1",
        seed=0,
        tsdf="official bounded: depth_trunc=2*estimated_radius; voxel=depth_trunc/256; sdf_trunc=5*voxel",
    )
    output.mkdir(parents=True, exist_ok=True)
    settings_path = output / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError(
            "Cached 2DGS settings differ; use a separate preprocessing output directory"
        )
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    manifest = stage_colmap(
        dataset_root, output / "input", settings["resolution"], background=background
    )
    status_path = output / "status.json"
    status = (
        json.loads(status_path.read_text())
        if status_path.exists()
        else dict(settings=settings, stages={})
    )
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if status.get("input_fingerprint", fingerprint) != fingerprint:
        raise ValueError("2DGS preprocessing inputs changed; use a separate output directory")
    status["input_fingerprint"] = fingerprint
    model = output / "model"
    commands = dict(
        train=[
            "train.py",
            "-s",
            str(output / "input"),
            "-m",
            str(model),
            "--iterations",
            "7000",
            "-r",
            "1",
            "--lambda_dist",
            str(settings["lambda_dist"]),
            "--depth_ratio",
            "1.0",
            "--test_iterations",
            "-1",
            "--save_iterations",
            "7000",
            "--quiet",
        ],
        extract=[
            "render.py",
            "-m",
            str(model),
            "--iteration",
            "7000",
            "--mesh_res",
            "256",
            "--num_cluster",
            str(settings["num_cluster"]),
            "--depth_ratio",
            "1.0",
            "--skip_train",
            "--skip_test",
            "--quiet",
        ],
    )
    if background == "white":
        commands["train"].append("--white_background")
    _record_effective_arguments(source, commands["train"], output)
    mesh_path = model / "train/ours_7000/fuse_post.ply"
    if (
        status.get("mesh")
        and mesh_path.is_file()
        and _sha256(mesh_path) != status["mesh"]["sha256"]
    ):
        raise ValueError("Cached 2DGS coarse mesh checksum changed")
    for name, command in commands.items():
        if status["stages"].get(name, {}).get("status") == "complete":
            expected = (
                model / "point_cloud/iteration_7000/point_cloud.ply"
                if name == "train"
                else mesh_path
            )
            if not expected.is_file():
                raise ValueError(f"Completed 2DGS cache is missing {expected}")
            continue
        log = output / f"{name}.log"
        print(
            f"ExMesh paper initialization: 2DGS {name}; {len(manifest['views'])} views; log={log}",
            flush=True,
        )
        started = time.monotonic()
        with log.open("w") as stream:
            result = subprocess.run(
                [sys.executable, "-u", *command],
                cwd=source,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        status["stages"][name] = dict(
            status="complete" if result.returncode == 0 else "failed",
            seconds=time.monotonic() - started,
            command=command,
            log=str(log),
        )
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(
                f"2DGS {name} failed ({result.returncode}):\n{log.read_text(errors='replace')[-5000:]}"
            )
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles() or not np.isfinite(np.asarray(mesh.vertices)).all():
        raise ValueError("2DGS extraction produced an empty or invalid coarse mesh")
    target = Path(config["trainer"]["initial_mesh"]).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target != mesh_path:
        shutil.copyfile(mesh_path, target)
    status["mesh"] = dict(
        path=str(target),
        vertices=len(mesh.vertices),
        faces=len(mesh.triangles),
        sha256=_sha256(target),
    )
    extraction_log = (output / "extract.log").read_text(errors="replace")
    status["tsdf_effective"] = {}
    for key, label in (
        ("voxel_size", "voxel_size"),
        ("sdf_trunc", "sdf_trunc"),
        ("depth_trunc", "depth_truc"),
    ):
        match = re.search(rf"^{label}:\s*([0-9.eE+-]+)", extraction_log, re.MULTILINE)
        if match is None:
            raise ValueError(f"Missing actual TSDF parameter in extraction log: {label}")
        status["tsdf_effective"][key] = float(match.group(1))
    if not np.isclose(
        status["tsdf_effective"]["depth_trunc"] / status["tsdf_effective"]["voxel_size"], 256
    ):
        raise ValueError("Actual TSDF voxel/depth settings do not match resolution 256")
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(
        f"2DGS coarse mesh ready: {len(mesh.vertices)} vertices / {len(mesh.triangles)} faces",
        flush=True,
    )
    return status


def prepare_pgsr(config, *, dataset_root):
    """Run/cache independent PGSR initialization for ExMesh."""
    from msbench.core.config import Config

    config = Config(config).to_dict()
    options = config.get("preprocess", {})
    if options.get("type") != "exmesh_pgsr":
        raise ValueError(f"Unsupported preprocessing type: {options.get('type')}")

    target = Path(config["trainer"]["initial_mesh"]).resolve()
    if target.exists():
        print(f"ExMesh coarse mesh already exists at {target}, reusing it.", flush=True)
        return dict(mesh=dict(path=str(target)))

    dataset = Path(dataset_root).resolve()
    output = Path(options.get("output", "outputs/exmesh-preprocess/pgsr")).resolve()
    source = Path(options.get("source_repo", "../ExMesh/PGSR")).resolve()
    if not source.exists():
        source = Path("outputs/exmesh-preprocess/pgsr_source").resolve()
    if not (source / "train.py").is_file():
        candidate = Path("/scratch/Workspace/ExMesh/PGSR").resolve()
        if (candidate / "train.py").is_file():
            source = candidate
        else:
            raise FileNotFoundError(f"PGSR source not found: {source}")

    output.mkdir(parents=True, exist_ok=True)
    holdout = int(options.get("holdout", 0))
    iterations = int(options.get("iterations", 5000))
    resolution = int(options.get("resolution", 2))
    ncc_scale = float(options.get("ncc_scale", 0.5))
    voxel_size = float(options.get("voxel_size", 0.01))
    max_depth = float(options.get("max_depth", 5.0))
    num_cluster = int(options.get("num_cluster", 1))
    white_bg = bool(config["trainer"].get("white_background", False))

    settings = dict(
        dataset=str(dataset),
        source=str(source),
        holdout=holdout,
        iterations=iterations,
        resolution=resolution,
        ncc_scale=ncc_scale,
        voxel_size=voxel_size,
        max_depth=max_depth,
        num_cluster=num_cluster,
        use_depth_filter=False,
    )
    if white_bg:
        settings["white_background"] = True

    settings_path = output / "pgsr_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    work = output / "source"
    if not work.exists():
        shutil.copytree(
            source,
            work,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "build", "*.egg-info"),
        )
    input_dir = output / "input"
    input_dir.mkdir(exist_ok=True)
    for name in ("images", "mask", "cameras.npz"):
        src, dst = dataset / name, input_dir / name
        if not src.exists():
            continue
        if dst.exists() or dst.is_symlink():
            if not dst.is_symlink() or dst.resolve() != src:
                if dst.is_dir() and not dst.is_symlink():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
                dst.symlink_to(src, target_is_directory=src.is_dir())
        else:
            dst.symlink_to(src, target_is_directory=src.is_dir())

    sparse = dataset / "sparse/0"
    if not sparse.is_dir():
        sparse = dataset / "sparse"
    (input_dir / "sparse").mkdir(exist_ok=True)
    for src in sparse.iterdir():
        if src.suffix not in {".txt", ".bin"}:
            continue
        dst = input_dir / "sparse" / src.name
        if not dst.exists():
            dst.symlink_to(src)

    names = sorted(p.stem for p in (dataset / "images").glob("*.png"))
    split = dict(
        train=[n for i, n in enumerate(names) if not holdout or i % holdout],
        test=[n for i, n in enumerate(names) if holdout and i % holdout == 0],
    )
    (input_dir / "split.json").write_text(json.dumps(split, indent=2) + "\n")

    state_path = output / "pgsr_status.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else dict(settings=settings, split=split, stages={})
    )
    model = output / "model"
    commands = dict(
        train=[
            "train.py",
            "-s",
            str(input_dir),
            "-m",
            str(model),
            "--quiet",
            "-r",
            str(resolution),
            "--ncc_scale",
            str(ncc_scale),
            "--iterations",
            str(iterations),
        ],
        extract=[
            "render.py",
            "-m",
            str(model),
            "--quiet",
            "--iteration",
            str(iterations),
            "--num_cluster",
            str(num_cluster),
            "--voxel_size",
            str(voxel_size),
            "--max_depth",
            str(max_depth),
        ],
    )
    if white_bg:
        commands["train"].append("--white_background")

    mesh_ply = model / "mesh/tsdf_fusion_post.ply"
    for name, command in commands.items():
        if state.get("stages", {}).get(name, {}).get("status") == "complete" and mesh_ply.exists():
            continue
        log = output / f"{name}.log"
        print(
            f"PGSR {name}: {len(split['train'])} train / {len(split['test'])} held-out views; log={log}",
            flush=True,
        )
        start = time.monotonic()
        with log.open("w") as stream:
            result = subprocess.run(
                [sys.executable, "-u", *command], cwd=work, stdout=stream, stderr=subprocess.STDOUT
            )
        state["stages"][name] = dict(
            status="complete" if result.returncode == 0 else "failed",
            duration_seconds=time.monotonic() - start,
            command=command,
            log=str(log),
        )
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        if result.returncode:
            print(log.read_text(errors="replace")[-6000:], flush=True)
            raise RuntimeError(f"PGSR {name} failed with exit code {result.returncode}")

    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_ply))
    if not mesh.has_triangles() or not np.isfinite(np.asarray(mesh.vertices)).all():
        raise RuntimeError("PGSR did not produce a valid nonempty mesh")

    target.parent.mkdir(parents=True, exist_ok=True)
    if target != mesh_ply:
        shutil.copyfile(mesh_ply, target)
    state["mesh"] = dict(path=str(target), vertices=len(mesh.vertices), faces=len(mesh.triangles))
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(
        f"PGSR coarse mesh ready: {len(mesh.vertices)} vertices / {len(mesh.triangles)} faces",
        flush=True,
    )
    return state


def prepare_exmesh_initializer(config, *, dataset_root):
    from msbench.core.config import Config

    config = Config(config).to_dict()
    options = config.get("preprocess", {})
    ptype = options.get("type", "exmesh_2dgs")
    if ptype == "exmesh_pgsr":
        return prepare_pgsr(config, dataset_root=dataset_root)
    elif ptype == "exmesh_2dgs":
        return prepare_2dgs(config, dataset_root=dataset_root)
    else:
        raise ValueError(f"Unsupported preprocessing type: {ptype}")

