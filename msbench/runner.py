"""File-based subprocess orchestration without importing method runtimes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from msbench.core.config import Config, load_config
from msbench.methods import canonical_method_name

PROTOCOL_VERSION = 1
STAGES = ("train", "render", "eval", "profile", "inspect")


def method_name(cfg: dict[str, Any]) -> str:
    names = []
    for section in ("trainer", "adapter"):
        data = cfg.get(section, {})
        name = data.get("type", data.get("name"))
        if name:
            try:
                names.append(canonical_method_name(str(name)))
            except KeyError:
                # New methods are provided by packages installed in the worker.
                names.append(str(name))
    if not names:
        raise ValueError("Config must declare trainer.type or adapter.type.")
    if len(set(names)) != 1:
        raise ValueError(f"Trainer and adapter methods disagree: {names}")
    return names[0]


def load_runtime(path: Path, method: str) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or data.get("schema_version") != PROTOCOL_VERSION:
        raise ValueError("Runtime config must declare schema_version: 1.")
    methods = data.get("methods")
    if not isinstance(methods, dict) or method not in methods:
        raise ValueError(f"Runtime config has no Python environment for method '{method}'.")
    runtime = methods[method]
    if not isinstance(runtime, dict) or not isinstance(runtime.get("python"), str):
        raise ValueError(f"Runtime '{method}' must provide a Python executable path.")
    python = Path(runtime["python"]).expanduser()
    if not python.is_absolute():
        python = path.resolve().parent / python
    # Do not resolve symlinks: doing so bypasses a venv's interpreter prefix.
    python = python.absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"Python executable does not exist or is not executable: {python}")
    env = runtime.get("env", {})
    if not isinstance(env, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env.items()
    ):
        raise ValueError(f"Runtime '{method}'.env must map strings to strings.")
    return {"python": str(python), "env": env}


def write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def plan_run(config: Path, runtimes: Path, stages: list[str], workspace: Path) -> dict[str, Any]:
    if not stages or any(stage not in STAGES for stage in stages):
        raise ValueError(f"Select stages explicitly from: {', '.join(STAGES)}")
    if len(stages) != len(set(stages)):
        raise ValueError("Each stage may appear only once in a run.")
    cfg = load_config(config)
    method = method_name(cfg)
    runtime = load_runtime(runtimes, method)
    output = cfg.get("output", {}).get("dir")
    if not output:
        raise ValueError("Isolated runs require output.dir in the experiment config.")
    root = workspace.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Workspace directory does not exist: {root}")
    output_dir = Path(output).expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    return {
        "protocol_version": PROTOCOL_VERSION, "method": method,
        "config": cfg, "runtime": runtime, "stages": stages,
        "workspace": str(root), "output_dir": str(output_dir.absolute()),
    }


def run_plan(plan: dict[str, Any]) -> Path:
    """Run stages in order; every stage must return a matching result manifest."""
    run_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:12]
    record_dir = Path(plan["output_dir"]) / "benchmark_runs" / run_id
    record_dir.mkdir(parents=True)
    snapshot = record_dir / "config.yaml"
    Config(plan["config"]).dump(snapshot)
    report = {
        "protocol_version": PROTOCOL_VERSION, "run_id": run_id,
        "method": plan["method"], "status": "running", "stages": [],
        "config": str(snapshot),
        "config_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "workspace": plan["workspace"], "python": plan["runtime"]["python"],
        "started_at_unix": time.time(),
    }
    try:
        report["workspace_revision"] = subprocess.check_output(
            ["git", "-C", plan["workspace"], "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        report["workspace_dirty"] = bool(subprocess.check_output(
            ["git", "-C", plan["workspace"], "status", "--porcelain"], text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        report["workspace_revision"] = None
    report_path = record_dir / "run.json"
    write_json(report_path, report)
    child_env = os.environ.copy()
    child_env.update(plan["runtime"]["env"])
    try:
        for stage in plan["stages"]:
            request_path = record_dir / f"{stage}.request.json"
            response_path = record_dir / f"{stage}.result.json"
            log = record_dir / f"{stage}.log"
            request = {
                "protocol_version": PROTOCOL_VERSION, "run_id": run_id,
                "stage": stage, "method": plan["method"], "config": str(snapshot),
                "output_dir": plan["output_dir"],
                "workspace": plan["workspace"],
            }
            write_json(request_path, request)
            command = [plan["runtime"]["python"], "-I", "-m", "msbench.worker",
                       "--request", str(request_path), "--response", str(response_path)]
            entry = {"stage": stage, "status": "running", "command": command,
                     "log": str(log), "result": str(response_path)}
            report["stages"].append(entry)
            write_json(report_path, report)
            print(f"[{plan['method']}/{stage}] log: {log}", flush=True)
            with log.open("w") as stream:
                completed = subprocess.run(command, cwd=plan["workspace"], env=child_env,
                                           stdout=stream, stderr=subprocess.STDOUT, check=False)
            entry["returncode"] = completed.returncode
            entry["status"] = "failed"
            if completed.returncode:
                raise RuntimeError(f"Stage '{stage}' exited with {completed.returncode}; see {log}")
            if not response_path.is_file():
                raise RuntimeError(f"Stage '{stage}' returned no result manifest; see {log}")
            result = json.loads(response_path.read_text())
            for key in ("protocol_version", "run_id", "method", "stage"):
                if result.get(key) != request[key]:
                    raise RuntimeError(f"Stage '{stage}' returned mismatched {key}.")
            if result.get("status") != "complete":
                raise RuntimeError(f"Stage '{stage}' did not complete: {result.get('error')}")
            artifacts = result.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                raise RuntimeError(f"Stage '{stage}' returned no artifact records.")
            for artifact in artifacts:
                if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
                    raise RuntimeError(f"Stage '{stage}' returned an invalid artifact record.")
                path = Path(artifact["path"])
                if not path.is_absolute() or not path.exists():
                    raise RuntimeError(f"Stage '{stage}' returned a missing artifact: {path}")
            entry["status"] = "complete"
            write_json(report_path, report)
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_at_unix"] = time.time()
        write_json(report_path, report)
    return report_path
