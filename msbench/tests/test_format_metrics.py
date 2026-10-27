"""Tests for tools/format_metrics.py."""

from __future__ import annotations

import json

from tools.format_metrics import read_row, target_from_config


def test_dtu_rows_report_mesh_metrics_and_training_stats_only(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # A stale image-metrics file must not leak into the DTU row.
    (run_dir / "metrics.json").write_text(json.dumps({"aggregate": {"psnr_mean": 30.0}}))
    (run_dir / "mesh_metrics.json").write_text(json.dumps({"overall": 0.5}))
    (run_dir / "train_stats.json").write_text(
        json.dumps({"total_time_s": 120.0, "peak_gpu_memory_mib": 2048})
    )
    config = tmp_path / "scan24.yaml"
    config.write_text(
        f"""
dataset: {{name: dtu, type: dtu, root: data/dtu/scan24, scene: scan24}}
trainer: {{type: exmesh}}
output:
  dir: {run_dir}
  mesh_metrics_file: {run_dir / "mesh_metrics.json"}
"""
    )

    row = read_row(target_from_config(config))

    assert row.cd == 0.5
    assert row.psnr is None and row.ssim is None and row.lpips is None and row.fps is None
    assert row.training_time_s == 120.0 and row.training_memory_mib == 2048
    assert row.metrics_file.endswith("mesh_metrics.json")
