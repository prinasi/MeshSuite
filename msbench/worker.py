"""Versioned worker protocol; imported in the selected method environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from msbench.core.config import load_config
from msbench.runner import PROTOCOL_VERSION, STAGES, method_name, write_json

COMMANDS = {
    "train": ["train"], "render": ["render", "images"],
    "eval": ["eval", "images"], "profile": ["profile"], "inspect": ["inspect"],
}


def environment_info(method: str) -> dict[str, Any]:
    """Record dependency versions and GPU identity without serializing secrets."""
    import torch

    distribution = importlib.metadata.distribution("msbench")
    record = distribution.read_text("RECORD") or ""

    info = {
        "python": sys.executable, "python_version": platform.python_version(),
        "platform": platform.platform(), "worker_source": str(Path(__file__).resolve()),
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_record_sha256": hashlib.sha256(record.encode()).hexdigest(),
        "packages": {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions() if dist.metadata.get("Name")
        },
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "torch_cxx11_abi": getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
    if method == "mesh-splatting":
        from msbench.runtime import optional_operator_status

        info["optional_operators"] = optional_operator_status()
    return info


def artifact_paths(stage: str, cfg: dict[str, Any], output_dir: Path) -> list[Path]:
    """Resolve the existing CLI's artifact paths without normalizing checkpoints."""
    from msbench.cli.config import merged_section, metrics_file

    if stage == "train":
        return [output_dir / "train_stats.json", output_dir / "ckpt"]
    if stage == "render":
        render = merged_section(cfg, "render", nested="images")
        root = Path(render.get("output_dir") or render.get("dir") or output_dir)
        split = str(render.get("split", cfg.get("dataset", {}).get("split", "test")))
        return [root / "renders" / split / "manifest.json"]
    if stage == "eval":
        return [Path(metrics_file(cfg))]
    if stage == "profile":
        return [Path(cfg.get("profile", {}).get("output") or output_dir / "profile.json")]
    return [Path(cfg.get("inspect", {}).get("output") or
                 cfg.get("output", {}).get("stats_file") or output_dir / "stats.json")]


def collect_artifacts(paths: list[Path], *, started_ns: int | None = None) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        if not path.exists():
            raise RuntimeError(f"Stage returned successfully but required artifact is missing: {path}")
        if path.is_dir():
            files = sorted(item for item in path.rglob("*") if item.is_file())
            if not files:
                raise RuntimeError(f"Checkpoint directory is empty: {path}")
            records.append({"path": str(path.absolute()), "kind": "directory",
                            "file_count": len(files), "size_bytes": sum(p.stat().st_size for p in files)})
        else:
            if started_ns is not None and path.stat().st_mtime_ns < started_ns:
                raise RuntimeError(f"Stage did not update its result artifact: {path}")
            payload = path.read_bytes()
            if not payload:
                raise RuntimeError(f"Artifact is empty: {path}")
            if path.suffix == ".json":
                data = json.loads(payload)
                if not isinstance(data, dict):
                    raise RuntimeError(f"JSON artifact must be an object: {path}")
                if path.name == "manifest.json":
                    frames = data.get("frames")
                    if not isinstance(frames, list) or not frames or len(frames) != data.get("num_frames"):
                        raise RuntimeError(f"Render manifest has inconsistent frame count: {path}")
                    for frame in frames:
                        for key in ("render_path", "gt_path"):
                            if frame.get(key) and not Path(frame[key]).is_file():
                                raise RuntimeError(f"Render manifest references missing image: {frame[key]}")
            records.append({"path": str(path.absolute()), "kind": "file", "size_bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest()})
    return records


def execute(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported worker protocol version.")
    stage = request.get("stage")
    if stage not in STAGES:
        raise ValueError(f"Unsupported stage: {stage}")
    distribution = importlib.metadata.distribution("msbench")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct_url.get("dir_info", {}).get("editable"):
        raise RuntimeError(
            "Isolated workers require a non-editable msbench installation. "
            "Editable environments can share and overwrite CUDA .so files; "
            "install a separately built wheel in each method environment."
        )
    cfg = load_config(request["config"])
    if method_name(cfg) != request["method"]:
        raise ValueError("Request method does not match config.")
    output_dir = Path(request["output_dir"])
    if Path(cfg["output"]["dir"]).absolute() != output_dir.absolute():
        raise ValueError("Request output directory does not match config.")
    result = {key: request[key] for key in ("protocol_version", "run_id", "stage", "method")}
    result["environment"] = environment_info(request["method"])
    started = time.time()
    started_ns = time.time_ns()
    from msbench.cli.app import app

    try:
        app(args=[*COMMANDS[stage], "--config", request["config"]], standalone_mode=False)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    result["artifacts"] = collect_artifacts(artifact_paths(stage, cfg, output_dir), started_ns=started_ns)
    result.update(status="complete", elapsed_s=time.time() - started)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    try:
        result = execute(request)
    except BaseException as exc:
        result = {key: request.get(key) for key in ("protocol_version", "run_id", "stage", "method")}
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(args.response, result)
        traceback.print_exc()
        raise SystemExit(1) from exc
    write_json(args.response, result)


if __name__ == "__main__":
    main()
