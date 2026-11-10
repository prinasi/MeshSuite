#!/usr/bin/env python3
"""Merge per-group Unity deployment reports into one plain-text summary.

``batch_unity_eval.sh`` writes one ``unity_metrics_report.json`` per
method x engine deployment setting x dataset under
``<reports-root>/<method>/<condition>/<dataset>/``; this tool collects them into
one text file (an overview of dataset means, then a per-scene table for every
group) and a companion CSV holding the same rows at full precision.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Method order and display names follow the manuscript; the bracketed config id
# is printed next to each name so a variant is never mistaken for its base.
METHODS = {
    "triangle-splatting": "Triangle-Splatting",
    "triangle-splatting-game-engine": "Triangle-Splatting (game engine, train_game_engine.py)",
    "2dts": "2DTS",
    "2dts-opaque": "2DTS (opaque)",
    "diffsoup": "DiffSoup",
    "mesh-splatting": "MeshSplatting",
}
DATASETS = {
    "mipnerf360": "Mip-NeRF 360",
    "tandt": "Tanks & Temples",
    "nerf_synthetic": "NeRF-Synthetic",
}
# Manuscript order: Std. before Cust.
CONDITIONS = {
    "standard-engine": ("Std.", "standard engine deployment"),
    "custom-engine": ("Cust.", "custom engine deployment"),
}
# Scene order of single_unity_eval.sh (Mip-NeRF 360: outdoor, then indoor).
SCENE_ORDER = {
    "mipnerf360": ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"],
    "tandt": ["truck", "train"],
    "nerf_synthetic": ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"],
}

# (header, report key, decimals): the manuscript's rounding conventions.
COLUMNS = (
    ("PSNR", "psnr", 2),
    ("SSIM", "ssim", 3),
    ("LPIPS", "lpips_vgg", 3),
    ("FPS", "fps", 0),
    ("GPU P50 ms", "gpu_frame_p50_ms", 2),
    ("GPU P95 ms", "gpu_frame_p95_ms", 2),
    ("Alloc. MiB", "rendering_memory_mib", 1),
    ("Asset MiB", "asset_memory_mib", 1),
    ("Load ms", "load_to_ready_ms", 0),
    ("vs native PSNR", "de_facto_psnr", 2),
)
CSV_KEYS = (
    "psnr", "ssim", "lpips_vgg", "fps", "engine_frame_p50_ms",
    "gpu_frame_p50_ms", "gpu_frame_p95_ms", "rendering_memory_mib",
    "unity_allocated_mib", "unity_reserved_mib", "asset_memory_mib",
    "load_to_ready_ms", "de_facto_psnr", "de_facto_ssim", "de_facto_lpips_vgg",
    "views", "profile_runs",
)


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt(value: Any, digits: int) -> str:
    number = finite(value)
    if number is None:
        return "N/A"
    return f"{number:,.{digits}f}" if digits == 0 else f"{number:.{digits}f}"


def expected_scenes(config_root: Path, method: str, dataset: str) -> list[str]:
    directory = config_root / method / dataset
    present = {path.stem for path in directory.glob("*.yaml")} if directory.is_dir() else set()
    ordered = [scene for scene in SCENE_ORDER.get(dataset, []) if scene in present]
    return ordered + sorted(present - set(ordered))


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]) for i, cell in enumerate(cells)
        ) + " |"

    return [line(headers), "|" + "|".join("-" * (width + 2) for width in widths) + "|", *map(line, rows)]


def load_group(reports_root: Path, method: str, condition: str, dataset: str) -> dict[str, Any] | None:
    directory = reports_root / method / condition / dataset
    path = directory / "unity_metrics_report.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    info_path = directory / "run_info.json"
    payload["run_info"] = json.loads(info_path.read_text()) if info_path.is_file() else {}
    payload["path"] = path
    return payload


def scene_mean(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Macro mean over scenes, as tools/format_unity_metrics_report.py aggregates."""
    keys = {key for row in rows for key in row} - {"dataset", "scene", "method", "condition", "runtime", "views"}
    mean: dict[str, Any] = {"views": sum(int(row.get("views") or 0) for row in rows)}
    for key in keys:
        values = [value for row in rows if (value := finite(row.get(key))) is not None]
        mean[key] = statistics.fmean(values) if values else None
    return mean


def metric_cells(row: dict[str, Any]) -> list[str]:
    return [fmt(row.get(key), digits) for _, key, digits in COLUMNS]


def summarize(
    reports_root: Path,
    config_root: Path,
    methods: list[str],
    datasets: list[str],
    conditions: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    overview: list[list[str]] = []
    sections: list[str] = []
    csv_rows: list[dict[str, Any]] = []
    for method in methods:
        method_name = f"{METHODS.get(method, method)} [{method}]"
        for dataset in datasets:
            scenes = expected_scenes(config_root, method, dataset)
            for condition in conditions:
                short, long = CONDITIONS[condition]
                setting = f"{short} ({long})"
                if method == "diffsoup" and condition == "standard-engine":
                    setting += ", baked RGBA texture atlas with alpha cutout"
                title = f"{method_name} | {setting} | {DATASETS.get(dataset, dataset)}"
                sections += ["", "=" * len(title), title, "=" * len(title)]
                if not scenes:
                    sections.append(f"NOT APPLICABLE: no configs under {config_root / method / dataset}")
                    continue
                group = load_group(reports_root, method, condition, dataset)
                if group is None:
                    sections.append(f"MISSING: no report under {reports_root / method / condition / dataset}")
                    overview.append([method, short, dataset, f"0/{len(scenes)}", *["N/A"] * len(COLUMNS), "MISSING"])
                    continue
                # A scene without image metrics has no completed capture.
                by_scene = {
                    row["scene"]: row for row in group.get("scenes", []) if finite(row.get("psnr")) is not None
                }
                present = [scene for scene in scenes if scene in by_scene]
                missing = [scene for scene in scenes if scene not in by_scene]
                status = "complete" if not missing else f"INCOMPLETE, missing {','.join(missing)}"
                info = group["run_info"]
                sections.append(f"report: {group['path']}")
                sections.append(
                    "run: " + ", ".join(
                        f"{key}={info[key]}" for key in ("finished", "git_commit", "gpu", "profile_runtime")
                        if info.get(key)
                    ) if info else "run: (no run_info.json)"
                )
                rows = [[scene, *metric_cells(by_scene[scene])] for scene in present]
                # Recomputed over this dataset's expected scenes only, so a report
                # holding any other scene can never shift the mean.
                aggregate = scene_mean([by_scene[scene] for scene in present])
                extra = sorted(set(by_scene) - set(scenes))
                mean_label = f"Mean ({len(present)}/{len(scenes)})"
                rows.append([mean_label, *metric_cells(aggregate)])
                sections += table(["Scene", *(header for header, _, _ in COLUMNS)], rows)
                if extra:
                    sections.append(f"WARNING: report also holds {', '.join(extra)}; excluded from this table")
                if missing:
                    sections.append(f"WARNING: mean covers {len(present)} of {len(scenes)} scenes; missing: {', '.join(missing)}")
                overview.append([method, short, dataset, f"{len(present)}/{len(scenes)}", *metric_cells(aggregate), status])
                for scene, row in [*((scene, by_scene[scene]) for scene in present), ("MEAN", aggregate)]:
                    csv_rows.append({
                        "method": method, "setting": short, "condition": condition,
                        "dataset": dataset, "scene": scene,
                        **{key: row.get(key) for key in CSV_KEYS},
                    })
    head = ["Method", "Setting", "Dataset", "Scenes", *(header for header, _, _ in COLUMNS), "Status"]
    lines = ["OVERVIEW: dataset means (macro mean over scenes)", "", *(table(head, overview) if overview else ["(no groups)"])]
    return [*lines, "", "PER-SCENE TABLES", *sections], csv_rows


def protocol_lines(reports_root: Path) -> list[str]:
    return [
        "MeshSuite engine deployment metrics (Unity)",
        f"Generated: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        f"Source reports: {reports_root}",
        "",
        "Protocol (single source: tools/format_unity_metrics_report.py, per-group reports):",
        "- Std. = standard engine deployment: conventional opaque vertex-coloured Unity mesh;",
        "  DiffSoup Std. is its baked RGBA texture atlas with alpha cutout (unity_baked).",
        "- Cust. = custom engine deployment: method-specific Unity renderer.",
        "- Variants deploy through their base method's renderer: 2dts-opaque -> 2dts (solid triangles",
        "  in Cust.), triangle-splatting-game-engine -> triangle-splatting.",
        "- PSNR/SSIM/LPIPS: held-out test views vs ground truth, raw RGB code values, Gaussian SSIM,",
        "  full-frame LPIPS-VGG. Captures use explicit 2x MSAA.",
        "- FPS: end-to-end throughput from the standalone Player, 1000 / median over profile runs of",
        "  the pooled engine frame time P50 (CPU + GPU, no readback or file I/O).",
        "- GPU P50/P95 ms: FrameTimingManager GPU frame time; Alloc. MiB: Unity Profiler",
        "  graphics-driver allocated bytes (the manuscript's VRAM / Alloc column); Asset MiB:",
        "  serialized TriAsset package size; Load ms: load-to-ready time.",
        "- vs native PSNR: Unity frames vs the method's de facto (CUDA) renders, i.e. port fidelity;",
        "  N/A when <output.dir>/renders/test/renders is absent.",
        "- Mean rows: macro mean of per-scene values (FPS is the mean of per-scene FPS).",
        "- Rounding: PSNR 2 dp, SSIM/LPIPS 3 dp, FPS integer, ms 2 dp, MiB 1 dp; means are computed",
        "  from unrounded scene values. Full precision is in the companion CSV.",
        "",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("configs"))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--output", type=Path, required=True, help="Summary .txt path")
    parser.add_argument("--csv", type=Path, help="Full-precision CSV (default: --output with .csv)")
    args = parser.parse_args()

    conditions = [condition for condition in CONDITIONS if condition in args.conditions]
    body, csv_rows = summarize(args.reports_root, args.config_root, args.methods, args.datasets, conditions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join([*protocol_lines(args.reports_root), *body]) + "\n")
    csv_path = args.csv or args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "setting", "condition", "dataset", "scene", *CSV_KEYS])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Summary: {args.output}")
    print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
