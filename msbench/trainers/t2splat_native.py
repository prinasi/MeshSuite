"""T²-Splat training through its vendored upstream two-stage pipeline.

T²-Splat (Tang et al., NeurIPS 2026) trains in two stages, each an upstream
script in ``msbench.vendor.t2splat``:

1. ``train.py``: geometry. This is MeshSplatting optimization plus Safe RDT and
   Dynamic Topology Evolution ("Heartbeat") with visibility pruning.
2. ``train_face_texture.py``: residual per-face Mip-textures on the frozen mesh.
   It is skipped when ``trainer.texture.iterations`` is 0 (the DTU preset).

Upstream runs each stage as its own process (``scripts/run_pipeline.py``).
Here they run in-process, and the RNG state is reset before each stage, as a
fresh process would. Checkpoints are written to ``<run>/ckpt/point_cloud/
iteration_<N>``, where the final ``N`` is ``trainer.max_steps`` (geometry plus
texture iterations).

MeshSuite controls only the dataset protocol: images, resolution, split,
background and the DTU loader. Method hyperparameters are passed straight
through to upstream's argument parser, which rejects unknown keys.
"""

from __future__ import annotations

import contextlib
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

# Trainer keys owned by MeshSuite's dataset protocol / runner, not upstream flags.
_PROTOCOL_KEYS = {
    "type", "name", "native_loop", "images", "resolution", "eval_split", "llffhold",
    "dtu_eval_mode", "white_background", "foreground_training", "texture", "seed",
    "max_steps",
}

# Stage-II options forwarded to ``train_face_texture.train_face_texture``.
_TEXTURE_DEFAULTS: dict[str, Any] = {
    # Upstream configs/common.yaml ``texture:`` block (paper App. A.1).
    "iterations": 5000,
    "face_tex_res": 8,
    "legacy_residual": True,
    "use_mip_texture": True,
    "lr_face_texture": 0.0005,
    "lr_geometry": 0.0,
    "lr_sh": 0.004,
    "lambda_l1": 0.8,
    "lambda_ssim": 0.2,
    "lambda_tv": 0.01,
    "lambda_l2": 0.01,
    "lr_decay_start": 1000,
    "lr_decay_ratio": 0.1,
    "texture_adam_offload": False,
}


def texture_settings(trainer_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the Stage-II settings, rejecting unknown keys."""
    raw = trainer_cfg.get("texture") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("trainer.texture must be a mapping")
    unknown = sorted(set(raw) - set(_TEXTURE_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown T²-Splat trainer.texture keys: {', '.join(unknown)}")
    settings = {**_TEXTURE_DEFAULTS, **raw}
    if int(settings["iterations"]) < 0:
        raise ValueError("trainer.texture.iterations must be >= 0")
    if int(settings["iterations"]) and int(settings["face_tex_res"]) < 1:
        raise ValueError("trainer.texture.face_tex_res must be positive")
    return settings


def geometry_argv(
    trainer_cfg: Mapping[str, Any],
    *,
    dataset_root: str,
    model_path: Path,
    geometry_iterations: int,
    quiet: bool = False,
) -> list[str]:
    """Build the upstream ``train.py`` argv for the geometry stage."""
    from msbench.vendor.t2splat.train import build_parser

    argv = [
        "-s", str(Path(dataset_root).expanduser()),
        "-m", str(model_path),
        "--iterations", str(int(geometry_iterations)),
        # As upstream scripts/run_pipeline.py: save only the final geometry checkpoint.
        "--save_iterations", str(int(geometry_iterations)),
        # In-training test reports only log metrics; MeshSuite evaluates separately.
        "--test_iterations", "-1",
        "--final_only_monitoring",
    ]
    if trainer_cfg.get("images") is not None:
        argv += ["-i", str(trainer_cfg["images"])]
    if trainer_cfg.get("resolution") is not None:
        argv += ["-r", str(trainer_cfg["resolution"])]
    if trainer_cfg.get("dtu_eval_mode") is not None:
        argv += ["--dtu_eval_mode", str(trainer_cfg["dtu_eval_mode"])]
    if bool(trainer_cfg.get("white_background", False)):
        argv.append("--white_background")
    if bool(trainer_cfg.get("eval_split", True)):
        argv.append("--eval")
    if quiet:
        argv.append("--quiet")

    parser, *_ = build_parser()
    actions = {action.dest: action for action in parser._actions}
    for key in sorted(set(trainer_cfg) - _PROTOCOL_KEYS):
        value = trainer_cfg[key]
        if key in {"iterations", "test_iterations", "save_iterations", "final_only_monitoring"}:
            raise ValueError(f"trainer.{key} is set by MeshSuite; use trainer.max_steps instead")
        if key not in actions:
            raise ValueError(f"Unknown T²-Splat trainer option: {key}")
        if value is None:
            continue
        action = actions[key]
        if action.nargs == 0:  # store_true flag
            if bool(value):
                argv.append(f"--{key}")
            elif action.default:
                raise ValueError(f"trainer.{key} defaults to True upstream and cannot be disabled")
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        argv += [f"--{key}", *(_cli_value(v) for v in values)]
    return argv


def _cli_value(value: Any) -> str:
    # Upstream argparse infers int types from integer defaults (e.g. 30_000).
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _checkpoint_dir(model_path: Path, iteration: int) -> Path:
    return model_path / "point_cloud" / f"iteration_{iteration}"


def _has_checkpoint(model_path: Path, iteration: int) -> bool:
    return (_checkpoint_dir(model_path, iteration) / "point_cloud_state_dict.pt").is_file()


@contextlib.contextmanager
def _restore_stdout():
    """Undo upstream ``safe_state``, which replaces ``sys.stdout`` for good."""
    stdout = sys.stdout
    try:
        yield
    finally:
        sys.stdout = stdout


def use_upstream_cuda_allocator() -> None:
    """Expandable CUDA segments, as upstream run_pipeline.py sets for every stage.

    T²-Splat's topology edits reallocate multi-GB buffers; without this the
    caching allocator fragments (Mip-NeRF 360 runs out of memory on 12 GB GPUs).
    The environment variable covers fresh processes; the live setting covers an
    allocator MeshSuite has already initialized (e.g. for peak-memory stats).
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        import torch

        if torch.cuda.is_available() and "expandable_segments" in os.environ["PYTORCH_CUDA_ALLOC_CONF"]:
            torch.cuda.memory._set_allocator_settings(os.environ["PYTORCH_CUDA_ALLOC_CONF"])
    except (ImportError, AttributeError, RuntimeError):  # pragma: no cover
        pass


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass


def run_t2splat_native_config(
    *,
    trainer_cfg: dict,
    dataset_root: str,
    output_dir: Path,
    max_steps: int,
    quiet: bool,
) -> dict:
    """Train T²-Splat; resumes at stage granularity from existing checkpoints."""
    texture = texture_settings(trainer_cfg)
    texture_iterations = int(texture["iterations"])
    geometry_iterations = int(max_steps) - texture_iterations
    if geometry_iterations <= 0:
        raise ValueError(
            f"trainer.max_steps={max_steps} must exceed trainer.texture.iterations={texture_iterations}"
        )

    use_upstream_cuda_allocator()
    model_path = Path(output_dir) / "ckpt"
    model_path.mkdir(parents=True, exist_ok=True)
    stages: list[str] = []
    start = time.time()

    if _has_checkpoint(model_path, max_steps):
        print(f"Found final T²-Splat checkpoint {_checkpoint_dir(model_path, max_steps)}; nothing to train.")
        return _summary(max_steps, max_steps, 0.0, stages)

    from msbench.runtime import require_method_cuda

    require_method_cuda("t2splat")
    argv = geometry_argv(
        trainer_cfg,
        dataset_root=dataset_root,
        model_path=model_path,
        geometry_iterations=geometry_iterations,
        quiet=quiet,
    )
    if _has_checkpoint(model_path, geometry_iterations):
        print(f"Reusing T²-Splat geometry checkpoint {_checkpoint_dir(model_path, geometry_iterations)}.")
        start_step = geometry_iterations
    else:
        # Upstream cannot resume Stage I mid-run (topology and Heartbeat state).
        from msbench.vendor.t2splat.train import run_training

        start_step = 0
        with _restore_stdout():
            run_training(argv)
        stages.append("geometry")
        _release_cuda()

    if texture_iterations:
        with _restore_stdout():
            _run_texture_stage(argv, model_path, geometry_iterations, texture, quiet=quiet)
        stages.append("texture")
        _release_cuda()

    if not _has_checkpoint(model_path, max_steps):
        raise RuntimeError(f"T²-Splat did not write its final checkpoint iteration_{max_steps}")
    return _summary(max_steps, start_step, time.time() - start, stages)


def _run_texture_stage(
    geometry_args: list[str],
    model_path: Path,
    geometry_iterations: int,
    texture: Mapping[str, Any],
    *,
    quiet: bool,
) -> None:
    """Run upstream Stage II with the same dataset arguments as Stage I."""
    from msbench.vendor.t2splat.train import build_parser
    from msbench.vendor.t2splat.train_face_texture import train_face_texture
    from msbench.vendor.t2splat.utils.general_utils import safe_state

    parser, model_params, _, pipeline_params = build_parser()
    args = parser.parse_args(geometry_args)
    safe_state(quiet)  # upstream Stage II starts in a fresh process
    train_face_texture(
        dataset=model_params.extract(args),
        pipeline=pipeline_params.extract(args),
        load_iteration=int(geometry_iterations),
        face_tex_res_list=[int(texture["face_tex_res"])],
        phase_iters_list=[int(texture["iterations"])],
        progressive=False,
        lr_face_texture=float(texture["lr_face_texture"]),
        lr_geometry=float(texture["lr_geometry"]),
        lr_sh=float(texture["lr_sh"]),
        lambda_l1=float(texture["lambda_l1"]),
        lambda_ssim=float(texture["lambda_ssim"]),
        lambda_tv=float(texture["lambda_tv"]),
        lambda_l2=float(texture["lambda_l2"]),
        lr_decay_start=int(texture["lr_decay_start"]),
        lr_decay_ratio=float(texture["lr_decay_ratio"]),
        # Test-view reports only log metrics; MeshSuite evaluates separately.
        test_interval=int(texture["iterations"]) + 1,
        quiet=quiet,
        legacy_residual=bool(texture["legacy_residual"]),
        use_mip_texture=bool(texture["use_mip_texture"]),
        texture_adam_offload=bool(texture["texture_adam_offload"]),
    )


def _summary(max_steps: int, start_step: int, seconds: float, stages: list[str]) -> dict:
    trained = max(int(max_steps) - int(start_step), 0)
    return {
        "total_steps": int(max_steps),
        "start_step": int(start_step),
        "trained_steps": trained,
        "total_time_s": seconds,
        "avg_step_time_ms": seconds * 1000.0 / max(trained, 1),
        "stages": stages,
        "final_losses": {},
    }
