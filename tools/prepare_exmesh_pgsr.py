"""Independent PGSR initialization, with an explicit all-view or held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgsr-repo", type=Path, default=Path("../ExMesh/PGSR"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument(
        "--holdout", type=int, default=8, help="Every Nth image is held out; 0 uses all views"
    )
    args = parser.parse_args()
    source = args.pgsr_repo.resolve()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if args.holdout < 0 or args.holdout == 1:
        parser.error("holdout must be 0 or at least 2")
    if not (source / "train.py").is_file():
        parser.error(f"PGSR source not found: {source}")
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(
        dataset=str(dataset),
        source=str(source),
        holdout=args.holdout,
        iterations=5000,
        resolution=2,
        ncc_scale=0.5,
        voxel_size=0.01,
        max_depth=5.0,
        num_cluster=1,
        use_depth_filter=False,
    )
    if args.white_background:
        settings["white_background"] = True
    settings_path = output / "pgsr_settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise RuntimeError(
            "Existing PGSR output has different settings; use a separate output directory"
        )
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
                raise RuntimeError(f"Unexpected input path: {dst}")
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
        train=[n for i, n in enumerate(names) if not args.holdout or i % args.holdout],
        test=[n for i, n in enumerate(names) if args.holdout and i % args.holdout == 0],
    )
    if not split["train"]:
        raise ValueError("No PGSR training views")
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
            "2",
            "--ncc_scale",
            ".5",
            "--iterations",
            "5000",
        ],
        extract=[
            "render.py",
            "-m",
            str(model),
            "--quiet",
            "--iteration",
            "5000",
            "--num_cluster",
            "1",
            "--voxel_size",
            ".01",
            "--max_depth",
            "5.0",
        ],
    )
    if args.white_background:
        commands["train"].append("--white_background")
    for name, command in commands.items():
        if state["stages"].get(name, {}).get("status") == "complete":
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
            raise SystemExit(result.returncode)
    import numpy as np
    import open3d as o3d

    path = model / "mesh/tsdf_fusion_post.ply"
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_triangles() or not np.isfinite(np.asarray(mesh.vertices)).all():
        raise RuntimeError("PGSR did not produce a valid nonempty mesh")
    state["mesh"] = dict(path=str(path), vertices=len(mesh.vertices), faces=len(mesh.triangles))
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state["mesh"], indent=2), flush=True)


if __name__ == "__main__":
    main()
