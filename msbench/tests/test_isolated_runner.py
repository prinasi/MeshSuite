"""Subprocess contract, artifact integrity, and controller import isolation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest
import yaml

from msbench.runner import load_runtime, method_name, plan_run, run_plan
from msbench.worker import artifact_paths, collect_artifacts


def test_controller_imports_without_any_runtime_dependency():
    code = '''
import sys
class BlockRuntimeImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "typer", "scipy"} or fullname.startswith("msbench.vendor"):
            raise AssertionError("runtime imported: " + fullname)
sys.meta_path.insert(0, BlockRuntimeImports())
import msbench.runner
import msbench.cli.main
import msbench.cli.run
assert not any(name.startswith("msbench.renderers") for name in sys.modules)
'''
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_method_identity_allows_new_plugins_but_rejects_mismatches():
    assert method_name({"trainer": {"type": "my-method"}, "adapter": {"type": "my-method"}}) == "my-method"
    assert method_name({"adapter": {"type": "my_method"}}) == "my_method"
    assert method_name({"trainer": {"type": "d2ts"}, "adapter": {"type": "2DTS"}}) == "2dts"
    with pytest.raises(ValueError, match="disagree"):
        method_name({"trainer": {"type": "2dts"}, "adapter": {"type": "diffsoup"}})


def test_runtime_config_preserves_venv_python_symlinks(tmp_path):
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    path = tmp_path / "runtimes.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "methods": {"2dts": {"python": "venv/bin/python"}}}))
    assert load_runtime(path, "2dts")["python"] == str(python)
    with pytest.raises(ValueError, match="no Python environment"):
        load_runtime(path, "diffsoup")


def test_worker_artifacts_reject_missing_stale_and_broken_render_images(tmp_path):
    result = tmp_path / "manifest.json"
    with pytest.raises(RuntimeError, match="missing"):
        collect_artifacts([result])
    result.write_text(json.dumps({"num_frames": 1, "frames": [{"render_path": str(tmp_path / "absent.png")}]}))
    with pytest.raises(RuntimeError, match="missing image"):
        collect_artifacts([result])
    result.write_text("{}")
    with pytest.raises(RuntimeError, match="inconsistent frame count"):
        collect_artifacts([result])
    os.utime(result, (1, 1))
    with pytest.raises(RuntimeError, match="did not update"):
        collect_artifacts([result], started_ns=time.time_ns())


def test_artifact_locations_follow_nested_image_config(tmp_path):
    cfg = {"render": {"split": "test", "images": {"split": "train", "dir": str(tmp_path)}}}
    assert artifact_paths("render", cfg, tmp_path / "output") == [tmp_path / "renders/train/manifest.json"]


@pytest.mark.parametrize("failure", [None, "exit", "missing", "mismatch"])
def test_subprocess_records_success_and_failures(tmp_path, failure):
    # A separate executable emulates an installed worker; this checks actual
    # process launch, cwd/env/argv, response identity, and failure manifests.
    python = tmp_path / "worker-python"
    python.write_text(f'''#!{sys.executable}
import hashlib, json, os, pathlib, sys
args = sys.argv[1:]
assert args[:3] == ["-I", "-m", "msbench.worker"]
assert os.environ["WORKER_SENTINEL"] == "isolated"
request = json.loads(pathlib.Path(args[4]).read_text())
assert pathlib.Path.cwd() == pathlib.Path({str(tmp_path)!r})
failure = {failure!r}
if failure == "exit":
    sys.exit(9)
if failure == "missing":
    sys.exit(0)
result = {{key: request[key] for key in ("protocol_version", "run_id", "method", "stage")}}
result["status"] = "complete"
artifact = pathlib.Path(request["output_dir"]) / (request["stage"] + ".json")
artifact.write_text("{{}}")
result["artifacts"] = [{{"path": str(artifact), "kind": "file", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}}]
if failure == "mismatch":
    result["run_id"] = "wrong-run"
pathlib.Path(args[6]).write_text(json.dumps(result))
''')
    python.chmod(0o755)
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump({"trainer": {"type": "future-method"}, "output": {"dir": "out"}}))
    runtimes = tmp_path / "runtimes.yaml"
    runtimes.write_text(yaml.safe_dump({"schema_version": 1, "methods": {"future-method": {
        "python": str(python), "env": {"WORKER_SENTINEL": "isolated"},
    }}}))
    plan = plan_run(config, runtimes, ["train", "eval"], tmp_path)
    if failure:
        with pytest.raises(RuntimeError):
            run_plan(plan)
        manifests = list((tmp_path / "out/benchmark_runs").glob("*/run.json"))
        report = json.loads(manifests[0].read_text())
        assert report["status"] == "failed"
        assert len(report["stages"]) == 1
    else:
        report = json.loads(run_plan(plan).read_text())
        assert report["status"] == "complete"
        assert [stage["status"] for stage in report["stages"]] == ["complete", "complete"]
        assert report["config_sha256"]
