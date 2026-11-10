"""Tests for the merged engine deployment summary of batch_unity_eval.sh."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "summarize_deployment_metrics.py"


def _config(root: Path, method: str, dataset: str, scenes: list[str]) -> None:
    for scene in scenes:
        path = root / method / dataset / f"{scene}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")


def _report(root: Path, method: str, condition: str, dataset: str, rows: list[dict]) -> None:
    directory = root / method / condition / dataset
    directory.mkdir(parents=True, exist_ok=True)
    scenes = [{"dataset": dataset, "method": method, "condition": condition, **row} for row in rows]
    # A deliberately wrong aggregate: the summary must recompute the mean.
    payload = {"scenes": scenes, "aggregate": {"psnr": 99.0}}
    (directory / "unity_metrics_report.json").write_text(json.dumps(payload))
    (directory / "run_info.json").write_text(json.dumps({"git_commit": "abc1234", "gpu": "RTX 4090"}))


def _summarize(tmp_path: Path, *extra: str) -> tuple[str, list[dict]]:
    output = tmp_path / "summary.txt"
    subprocess.run(
        [sys.executable, str(TOOL), "--reports-root", str(tmp_path / "reports"),
         "--config-root", str(tmp_path / "configs"), "--output", str(output), *extra],
        check=True, cwd=REPO, capture_output=True,
    )
    with output.with_suffix(".csv").open() as handle:
        return output.read_text(), list(csv.DictReader(handle))


def test_means_cover_only_completed_scenes_of_the_dataset(tmp_path: Path) -> None:
    configs, reports = tmp_path / "configs", tmp_path / "reports"
    _config(configs, "2dts", "tandt", ["truck", "train"])
    _report(reports, "2dts", "custom-engine", "tandt", [
        {"scene": "truck", "psnr": 22.0, "ssim": 0.8, "lpips_vgg": 0.2, "fps": 100.0},
        {"scene": "train", "psnr": 20.0, "ssim": 0.7, "lpips_vgg": 0.3, "fps": 50.0},
        {"scene": "chair", "psnr": 35.0, "fps": 900.0},  # another dataset's scene
    ])
    text, rows = _summarize(tmp_path, "--methods", "2dts", "--datasets", "tandt")

    mean = next(row for row in rows if row["setting"] == "Cust." and row["scene"] == "MEAN")
    assert float(mean["psnr"]) == 21.0
    assert float(mean["fps"]) == 75.0
    mean_line = next(line for line in text.splitlines() if line.startswith("| Mean (2/2)"))
    assert [cell.strip() for cell in mean_line.strip("|").split("|")][:5] == ["Mean (2/2)", "21.00", "0.750", "0.250", "75"]
    assert "report also holds chair" in text
    # Std. was never run, so it is reported missing rather than silently absent.
    assert "MISSING: no report" in text
    assert "99.00" not in text


def test_flags_incomplete_groups_and_methods_without_configs(tmp_path: Path) -> None:
    configs, reports = tmp_path / "configs", tmp_path / "reports"
    _config(configs, "diffsoup", "tandt", ["truck", "train"])
    _report(reports, "diffsoup", "standard-engine", "tandt", [
        {"scene": "truck", "psnr": 21.5},
        {"scene": "train", "psnr": None},  # listed in the report but never scored
    ])
    text, _ = _summarize(
        tmp_path, "--methods", "diffsoup", "2dts-opaque",
        "--datasets", "tandt", "nerf_synthetic", "--conditions", "standard-engine",
    )

    assert "INCOMPLETE, missing train" in text
    assert "WARNING: mean covers 1 of 2 scenes; missing: train" in text
    assert "baked RGBA texture atlas" in text
    assert "2DTS (opaque) [2dts-opaque] | Std. (standard engine deployment) | NeRF-Synthetic" in text
    assert "NOT APPLICABLE: no configs" in text


def test_batch_script_parses_and_summarizes_only(tmp_path: Path) -> None:
    script = REPO / "batch_unity_eval.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    result = subprocess.run(
        ["bash", str(script), "--summary-only", "--eval-root", str(tmp_path),
         "--methods", "mesh-splatting", "--datasets", "tandt", "--python", sys.executable],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    summary = (tmp_path / "deployment_metrics_summary.txt").read_text()
    assert "MeshSplatting [mesh-splatting] | Cust. (custom engine deployment) | Tanks & Temples" in summary
    assert "Summary:" in result.stdout
