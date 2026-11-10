<!-- @format -->

# MeshSuite: A Unified Benchmark for Triangle- and Mesh-Based Neural Rendering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2609.01306-b31b1b.svg)](https://arxiv.org/abs/2609.01306)
[![Project Page](https://img.shields.io/badge/Project-Page-2563eb.svg)](https://x-up-lab.github.io/research/meshsuite)

**MeshSuite** is a unified, Python-first benchmarking toolkit and evaluation framework for comparing triangle-splatting-style radiance field reconstruction methods, bridging academic novel view synthesis research and real-time industrial game engine deployment (Unity). Visit the [Project Page](https://x-up-lab.github.io/research/meshsuite) for interactive comparisons and full leaderboards.

## Overview

MeshSuite provides a single, consistent evaluation framework for comparing different triangle-based radiance field reconstruction methods. Instead of wrangling per-repo scripts and ad-hoc metrics, you get:

- **Standardized metrics** -- PSNR, SSIM, LPIPS, and geometry quality, computed the same way for every method.
- **Adapter pattern** -- plug any triangle-splatting method into the same API.
- **Reproducible experiments** -- YAML configs lock down every hyperparameter.
- **CLI-first workflow** -- inspect, profile, evaluate, and compare from the terminal.
- **Industrial Engine Parity** -- export method-preserving `.triasset` packages and evaluate real-time framerates and rendering fidelity natively in Unity.

## Supported Methods

| Method                       | Adapter                    | Status                                   |
| ---------------------------- | -------------------------- | ---------------------------------------- |
| 2D Triangle Splatting (2DTS) | `D2TSAdapter`              | Config/train/render/eval/export support  |
| Triangle Splatting           | `TriangleSplattingAdapter` | Config/train/render/eval/profile/export support |
| MeshSplatting                | `MeshSplattingAdapter`     | Config/train/render/eval/profile/export support |
| DiffSoup                     | `DiffSoupAdapter`          | Config/train/render/eval/export support; In-shader Micro-MLP support |
| ExMesh                       | `ExMeshAdapter`            | Native training/render/eval/profile/export; DTU and NeRF-Synthetic configs; beyond the paper |
| T²-Splat                     | `T2SplatAdapter`           | Native two-stage training/render/eval/profile/mesh export; Mip-NeRF 360, Tanks & Temples and DTU configs; beyond the paper |

### ExMesh

ExMesh goes beyond the MeshSuite paper (arXiv v5), which does not evaluate it; its
results are not part of the paper's tables.

ExMesh uses the same dataset configs as every other method (e.g. DTU from
`configs/base/dtu.yaml`: foreground evaluation on white, 1/2 resolution, all views
used for training), so its results are directly comparable:

```bash
bash single_train.sh exmesh dtu/all 0
bash single_train.sh exmesh nerf_synthetic/all 0
```

ExMesh is benchmarked only on the object-centric datasets (DTU and NeRF-Synthetic), as
in its paper. It refines one bounded, textured mesh from a coarse initialization, using
object masks and a fixed face budget, so it does not apply to unbounded Mip-NeRF 360 and
Tanks & Temples scenes; MeshSuite ships no ExMesh configs for them.

On NeRF-Synthetic its method settings follow the ExMesh paper: 8k iterations and a
coarse initial mesh from 2DGS (7k iterations, TSDF at 256³) built on the training
split, as the authors confirmed; the pinned 2DGS source is cloned to
`outputs/exmesh-preprocess/2dgs_source` on first use, and its
`diff_surfel_rasterization` extension must be installed. NeRF-Synthetic is trained
without depth supervision, as in the paper.

On DTU the paper's setup reaches only mean CD 0.98 on the shared protocol (paper:
0.58), so MeshSuite uses the best-performing settings it found instead
(`configs/base/exmesh-dtu.yaml`): the PGSR initializer bundled with the official
ExMesh repo, extracted with its depth filter at TSDF voxel 0.003 and decimated to 30k
faces; the repo's 10k iterations; and a 16x offset (`lambda_deviation`) weight.
Per-scene `lambda_smooth`/`lambda_depth` and the LR schedule come from the official
`scripts/run_dtu.py`. The pinned ExMesh source is cloned to
`outputs/exmesh-preprocess/exmesh_source` on first use, and PGSR's
`diff_plane_rasterization` extension must be installed. ExMesh on DTU also reads
monocular depth priors from `data/dtu/<scan>/mono_priors/da3/`.

The initialization is built automatically before training, on the same views as
ExMesh training, and cached under `outputs/exmesh-preprocess/`. ExMesh needs the
`exmesh` extra (`pip install -e ".[exmesh]"`).

### T²-Splat

[T²-Splat](https://github.com/t2-splat/T2Splat) (adaptive-topology mesh splatting
with texture residuals, NeurIPS 2026) goes beyond the MeshSuite paper (arXiv v5),
which does not evaluate it.

T²-Splat extends MeshSplatting. Stage I optimizes a connected mesh: Safe RDT
remeshing, Dynamic Topology Evolution (edge collapse / vertex split) and visibility
pruning. Stage II freezes the mesh and fits residual per-face Mip-textures. The
upstream code (commit `b1b44d1`) is vendored under `msbench/vendor/t2splat`, and its
CUDA rasterizer is built as the `t2splat` backend.

It uses the same dataset configs as every other method (for example, DTU: all
views, 1/2 resolution, foreground on white). Its method settings reproduce the
authors' per-scene repository presets (upstream `configs/common.yaml` and
`configs/<dataset>.yaml`):

- Mip-NeRF 360 and Tanks & Temples: 30k geometry and 5k texture iterations
  with 8×8 Mip-textures.
- DTU: the 20k-iteration, geometry-only per-scene best preset.

`msbench/tests/test_t2splat.py` checks every scene config against the options
that upstream's `scripts/run_pipeline.py` resolves. No method uses external
normal/depth priors. On DTU, T²-Splat's own mesh is scored after removing small
components, as upstream `create_ply.py --clean --geometry-only` does. As in the
paper, there are no NeRF-Synthetic configs.

Reproduced results match the paper: Mip-NeRF 360 bicycle 23.95 PSNR / 0.705 SSIM
(paper 23.96 / 0.709), Tanks & Temples truck 22.94 / 0.836 (22.91 / 0.836), and DTU
scan24 Chamfer distance 0.54 (0.56). MeshSuite's LPIPS (VGG, inputs scaled to
[-1, 1], as for every method) is about 0.03 higher than the paper's, which comes
from upstream's 3DGS-style metric fed [0, 1] images: on the same renders, bicycle
0.277 vs 0.249 and truck 0.197 vs 0.166 (paper 0.247 and 0.167).

The Mip-NeRF 360 presets need a GPU with about 24 GB (bicycle peaks at 24 GB; garden
runs out of memory on 12 GB in the restricted Delaunay step). DTU and truck fit in
12 GB.

```bash
pip install -e ".[t2splat]" && pip install --no-build-isolation ./submodules/effrdel
MSBENCH_CUDA_BACKENDS=t2splat,simple-knn python setup.py build_ext --inplace
bash single_train.sh t2splat dtu/all 0
bash single_train.sh t2splat mipnerf360/all 0
```

### Triangle Splatting for game engines

The optional `--train_game_engine` flag (method id `triangle-splatting-game-engine`)
trains Triangle Splatting with the official repository's `train_game_engine.py`,
which produces opaque, solid triangles for import into a game engine. After
densification, an extra loss pushes triangles towards opaque and sharp ones, and
pruning also drops transparent triangles (opacity ≤ 0.1, then ≤ 0.7 after 29K
iterations) and soft ones (σ ≥ 1 after 28K). Every other setting, including the
dataset protocol, comes from the matching `configs/triangle-splatting` scene config.
Outputs go to `outputs/triangle-splatting-game-engine/<dataset>/<scene>`.

The flag has no scene configs of its own. Each run uses the `configs/triangle-splatting`
scene config followed by `configs/base/triangle-splatting-game-engine-<dataset>.yaml`,
which adds the recipe in `configs/base/triangle-splatting-game-engine.yaml`. Mip-NeRF
360, Tanks & Temples and NeRF-Synthetic have that overlay. The method id names only
the output tree and reports and is not accepted as a method argument.

```bash
bash single_train.sh triangle-splatting mipnerf360/all 0 --train_game_engine
bash single_unity_eval.sh triangle-splatting mipnerf360/all 0 --standard-engine --train_game_engine
```

`single_unity_eval.sh`, `single_export_unity.sh` and `single_unity_video.sh`
accept `--train_game_engine` too. They deploy the variant with the
`triangle-splatting` package format and renderers, and keep its reports under
its own name.

## Installation

A normal install now provides the lightweight benchmark controller and does **not**
compile CUDA, even when `nvcc` or `CUDA_HOME` is present. Native commands use the
`runtime` extra. Existing command/config/script interfaces are retained.

### Controller environment

```bash
python -m pip install .
msbench run --help
```

The controller needs Python >= 3.10 and PyYAML; it does not import PyTorch,
CUDA extensions, or method adapters.

### Method environments

Create/manage each environment yourself, installing the PyTorch/torchvision and
CUDA toolkit versions appropriate to that method. For example, in the 2DTS environment:

```bash
# Install Python dependencies/build tools first; this does not compile CUDA.
python -m pip install '.[runtime,method-2dts,cuda]'

# Compile only 2DTS and its shared KNN dependency using this environment's torch.
MSBENCH_CUDA_BACKENDS=2dts python -m pip install '.[runtime,method-2dts,cuda]' --no-build-isolation
```

Use `method-triangle-splatting`, `method-mesh-splatting`, `method-diffsoup`, or
`t2splat` with the corresponding `MSBENCH_CUDA_BACKENDS` value for the other methods.
ExMesh compiles nothing in MeshSuite; install its `exmesh` extra instead.
Add `[geometry]` when running optional geometric evaluation. MeshSplatting
configs enabling restricted Delaunay additionally require a separately installed
`submodules/effrdel` (`rdel`) in that method environment.

For isolated runs, use **non-editable** installs so each environment owns its
compiled operators. Editable installs remain supported for development with the
existing native CLI/scripts; isolated workers reject them because multiple
editable environments can share and overwrite the same `.so` files.
Setuptools build directories are separated by interpreter, selected backends,
PyTorch/CUDA versions, and architecture settings.

CUDA names: `2dts`, `triangle-splatting`, `mesh-splatting`, `t2splat`, `diffsoup`, `simple-knn`.
Selecting one of the first four automatically includes `simple-knn`. Unknown
names fail the build. To compile every bundled backend explicitly:

```bash
MSBENCH_BUILD_CUDA=1 python -m pip install '.[runtime,cuda]' --no-build-isolation
# Equivalent selection: MSBENCH_CUDA_BACKENDS=all
```

| Environment variable | Meaning | Default |
| --- | --- | --- |
| `MSBENCH_SKIP_CUDA` | Always skip compilation; overrides build requests | Unset |
| `MSBENCH_BUILD_CUDA` | Explicitly request all bundled extensions | Unset |
| `MSBENCH_CUDA_BACKENDS` | Select methods/operators, automatically including KNN | No compilation |
| `MSBENCH_CUDA_ARCHS` | CUDA architectures for DiffSoup's CMake build | `89` |
| `TORCH_CUDA_ARCH_LIST` | CUDA architectures for PyTorch-built extensions | PyTorch default |
| `MSBENCH_CCACHE` | Use ccache through PyTorch/CMake compiler launchers | Unset |
| `MAX_BUILD_JOBS` | Compilation job limit | Auto |

Install build tools and PyTorch **before** using `--no-build-isolation`; pip does
not manage build dependencies in that mode. See the [pip build-system documentation](https://pip.pypa.io/en/stable/reference/build-system/).

The existing `environment.yml` and `requirements.txt` remain convenience options
for an all-method development environment. CUDA compilation is now a separate,
explicit step. They are not version locks for reproducible published runs.

### All-method development environment

```bash
git clone https://github.com/prinasi/MeshSuite.git
cd MeshSuite
conda env create -f environment.yml
conda activate msbench

# Use this environment's CUDA development toolkit, matching PyTorch cu118.
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CUDA_HOME/bin/nvcc"

# Build for the visible GPU (e.g. RTX 3060: 8.6 / 86).
export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')"
export MSBENCH_CUDA_ARCHS="${TORCH_CUDA_ARCH_LIST/./}"
MSBENCH_BUILD_CUDA=1 python -m pip install -e ".[dev,cuda]" --no-build-isolation
```

In an existing environment with PyTorch (>= 2.0 with CUDA support), install the
matching CUDA compiler and the build dependencies first:

```bash
# For PyTorch cu118, install the CUDA 11.8 compiler and development headers.
# The runtime-only `cudatoolkit` package does not provide nvcc.
conda install -c nvidia/label/cuda-11.8.0 cuda-nvcc=11.8 cuda-cudart-dev=11.8 cuda-cccl=11.8 cuda-driver-dev=11.8 cuda-nvtx=11.8
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"

export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')"
export MSBENCH_CUDA_ARCHS="${TORCH_CUDA_ARCH_LIST/./}"

# --no-build-isolation requires build dependencies to be installed first.
python -m pip install "setuptools>=68" wheel "nanobind>=2.11.0" "cmake>=3.20" ninja
python -m pip install -r requirements.txt

MSBENCH_BUILD_CUDA=1 python -m pip install -e ".[dev,cuda]" --no-build-isolation
```

> **Tip for Multi-CUDA Systems**:
> Match the development toolkit to `python -c 'import torch; print(torch.version.cuda)'`; “11.8 or newer” alone is insufficient for PyTorch cu118. Set both `CUDA_HOME` and `CUDACXX` so PyTorch and CMake (used by DiffSoup) use the same compiler. For a system toolkit, replace `CUDA_HOME` with its actual installation directory. Without a visible GPU, set the architecture variables manually (e.g. `8.6` / `86` for RTX 3060, `8.9` / `89` for RTX 4090).
>
> To retain these settings across Conda activations, run `conda env config vars set CUDA_HOME="$CUDA_HOME" CUDACXX="$CUDACXX" TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" MSBENCH_CUDA_ARCHS="$MSBENCH_CUDA_ARCHS"`, then reactivate the environment.

### Isolated benchmark execution

Copy [configs/runtimes.example.yaml](configs/runtimes.example.yaml), replacing each
interpreter path with your method environment's Python. No environments are created
or modified by the runner.

```bash
msbench run --config configs/triangle-splatting/mipnerf360/bicycle.yaml \
  --runtimes /path/to/runtimes.yaml \
  --stage train --stage render --stage eval --stage profile --dry-run

# Remove --dry-run to execute the stages in the stated order.
```

The equivalent lightweight entry point is `msbench-run`. Relative dataset/output
paths use the current working directory, or the directory supplied by `--workspace`.
Workers launch with `python -I` and load the package installed in their own environment.
The existing native outputs keep their paths. Additional run records go to:

```text
outputs/{method}/{dataset}/{scene}/benchmark_runs/{run_id}/
  config.yaml
  run.json
  {stage}.request.json
  {stage}.result.json
  {stage}.log
```

Supported isolated stages are `train`, `render`, `eval`, `profile`, and `inspect`.
A stage fails when its worker fails, its response identity does not match, or its
required result artifacts are missing/stale. Existing Unity, viewer, connectivity,
video, and export commands remain available through their current entry points.

Training requires a working CUDA device and loadable method-specific rasterizer/KNN
operators. KNN never falls back silently to `torch.cdist`. Restricted Delaunay's
`rdel` dependency is checked when enabled; T²-Splat always runs it and also rejects an
effrdel build older than this checkout's `submodules/effrdel` (no `rdel.run`
`face_selector`/`return_debug`). MeshSplatting retains its existing optional
fused-SSIM choice and records its availability. Evaluation still runs in the method
environment to preserve the current metric semantics; shared image scoring is a
separate next phase.

See [BENCHMARK_MAINTENANCE.md](BENCHMARK_MAINTENANCE.md) for the code audit,
method integration contract, limitations, and proposed maintenance roadmap.

## Dataset Setup

MeshSuite evaluates methods on standard novel view synthesis benchmarks: **Mip-NeRF 360**, **NeRF-Synthetic (Blender)**, **Tanks & Temples**, and **DTU**.

Readers should download the datasets from their official sources and place or symlink them into the `data/` directory according to the default format:

```bash
mkdir -p data
ln -s /path/to/MipNeRF360 data/mipnerf360
ln -s /path/to/nerf_synthetic data/nerf_synthetic
ln -s /path/to/tandt data/tandt
ln -s /path/to/DTU data/dtu
```

Each scene directory is expected to look like this:

```text
data/
├── mipnerf360/<scene>/       images/ (images_2, images_4, ...), sparse/0/
├── nerf_synthetic/<scene>/   train/, test/, transforms_{train,test}.json
├── tandt/<scene>/            images/, sparse/0/
└── dtu/
    ├── ObsMask/, Points/stl/ # official DTU evaluation data (Chamfer distance)
    └── scan<id>/             images/, mask/, sparse/0/, cameras.npz
                              # ExMesh also needs mono_priors/da3/
```

The 15 DTU evaluation scans are 24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110,
114, 118, and 122. Config paths are lowercase (`data/dtu`), so the symlink name
must match on case-sensitive file systems.

DTU is evaluated as a mesh benchmark. Every method trains on all views of each scan,
as in the methods' papers, and only DTU mesh metrics (Chamfer distance) are reported;
`single_train.sh` skips image rendering and NVS metrics on DTU.

## Quick Start

```bash
# Train, render, evaluate, inspect, and profile from one experiment config
msbench train --config configs/triangle-splatting/mipnerf360/bicycle.yaml
msbench render images --config configs/triangle-splatting/mipnerf360/bicycle.yaml
msbench eval images --config configs/triangle-splatting/mipnerf360/bicycle.yaml
msbench inspect --config configs/triangle-splatting/mipnerf360/bicycle.yaml
msbench profile --config configs/triangle-splatting/mipnerf360/bicycle.yaml

# Config values can still be overridden from the CLI
msbench render video --config configs/triangle-splatting/mipnerf360/bicycle.yaml --output-dir outputs/demo-video

# Train complete datasets with the scene pipeline helper
bash single_train.sh triangle-splatting mipnerf360/all 0
bash single_train.sh triangle-splatting tandt/all 0
bash single_train.sh triangle-splatting dtu/all 0
bash single_train.sh mesh-splatting mipnerf360/all 0
bash single_train.sh mesh-splatting tandt/all 0
bash single_train.sh mesh-splatting dtu/all 0
bash single_train.sh 2dts mipnerf360/all 0
bash single_train.sh 2dts tandt/all 0
bash single_train.sh 2dts dtu/all 0

# Export feature-preserving Unity-native assets from completed runs
bash single_export_unity.sh triangle-splatting mipnerf360/all 0
bash single_export_unity.sh mesh-splatting mipnerf360/garden 0

# Export missing Unity assets, capture Unity frames, profile FPS/memory, and format report tables
bash single_unity_eval.sh 2dts mipnerf360/bicycle 0 --unity "$UNITY" --unity-project unity
```

## Profiling

`msbench profile` measures the rendering performance of a trained checkpoint
and writes a versioned JSON report. It reuses the same config resolution as
train/render/eval, so a completed run can be profiled with the same config:

```bash
msbench profile --config configs/triangle-splatting/mipnerf360/bicycle.yaml
```

The defaults in `configs/base/_base_.yaml` write the report to
`outputs/{method}/{dataset}/{scene}/profile.json` and measure the `test` split
with 100 timed repeats after 5 untimed warmup renders.

### Measured metrics

| Metric | Top-level key | Notes |
| ------ | ------------- | ----- |
| Forward latency | `forward_latency_ms` | Mean of the timed eval renders |
| FPS | `fps` | `1000 / forward_latency_ms` |
| Peak CUDA memory | `peak_cuda_memory_gb` | Peak device allocation during one eval render |
| Primitive count | `primitive_count` | Triangles/faces/primitives in the checkpoint |
| Checkpoint size | `checkpoint_size_mb` | On-disk state-dict size |
| Backward latency | `backward_latency_ms` | Mean train-mode backward time; opt-in |

The `forward` object (and `backward`, when enabled) carries the full timing
summary (`mean_ms`, `std_ms`, `p50_ms`, `p95_ms`, `min_ms`, `max_ms`, `fps`),
and `memory` reports both allocated and reserved peaks. Timings use CUDA events
when the adapter runs on a CUDA device and wall-clock time otherwise. The
report shape is versioned under `schema_version`.

### Options and config keys

Explicit command-line flags override the config; otherwise the `profile:`
block from `--config` is used:

```yaml
profile:
  split: test
  repeats: 100
  warmup: 5
  backward: false
  output: outputs/{method}/{dataset}/{scene}/profile.json
```

| Flag | Description |
| ---- | ----------- |
| `--method/-m`, `--checkpoint/-c`, `--dataset/-d` | Override adapter method, checkpoint path, or dataset root |
| `--scene` | Scene name; resolves `{scene}` path templates |
| `--split/-s` | Dataset split (train/test/all) |
| `--repeats/-r` | Number of timed repetitions |
| `--warmup/-w` | Number of untimed warmup renders |
| `--backward/--no-backward` | Also measure backward-pass latency |
| `--output/-o` | JSON report path |

Fast smoke run on the same config:

```bash
msbench profile \
  --config configs/triangle-splatting/mipnerf360/bicycle.yaml \
  --repeats 3 \
  --warmup 1 \
  --backward \
  --output /tmp/bicycle-profile.json
```

### Example report

Triangle Splatting `bicycle` (4.89 M triangles, RTX 4090):

```json
{
  "schema_version": 1,
  "method": "triangle-splatting",
  "dataset": "data/mipnerf360/bicycle",
  "split": "test",
  "device": "cuda:0",
  "camera_name": "_DSC8679",
  "repeats": 100,
  "warmup": 5,
  "status": "complete",
  "forward_latency_ms": 14.847,
  "fps": 67.35,
  "peak_cuda_memory_gb": 3.399,
  "primitive_count": 4888414,
  "checkpoint_size_mb": 1100.22,
  "forward": {
    "mean_ms": 14.847,
    "std_ms": 0.151,
    "p50_ms": 14.850,
    "p95_ms": 15.037,
    "min_ms": 14.557,
    "max_ms": 15.632,
    "fps": 67.35
  },
  "memory": {
    "peak_memory_allocated_gb": 3.399,
    "peak_memory_reserved_gb": 3.697
  }
}
```

Running with `--backward` adds a `backward` summary with the same shape plus
`backward_latency_ms` (about 27.7 ms for this scene).

### CUDA extension rebuilds

Native rasterization requires the bundled CUDA extensions. Rebuild them
whenever the active PyTorch/CUDA environment changes; an ABI mismatch surfaces
as an extension import error when the adapter loads:

```bash
conda activate msbench
MSBENCH_BUILD_CUDA=1 \
  MSBENCH_CUDA_BACKENDS=triangle-splatting,simple-knn \
  python setup.py build_ext --inplace --force
```

Set `MSBENCH_CUDA_BACKENDS` to the backends you need and `MSBENCH_CUDA_ARCHS`
to your GPU architecture (default `89`); set `MSBENCH_SKIP_CUDA=1` to skip
extension builds entirely.

## Unity-native Evaluation

MeshSuite can export method-preserving Unity `.triasset` packages, render Unity
test-view images for PSNR/SSIM/LPIPS, and profile FPS, GPU latency and memory through a
Linux standalone Development Player. The workflow is designed for completed
config-driven runs under `outputs/{method}/{dataset}/{scene}`.

Unity replays cameras from COLMAP binaries. On NeRF-Synthetic, the wrapper first writes
a derived scene under `outputs/unity_replay/nerf_synthetic/<scene>` (`UNITY_REPLAY_ROOT`
overrides the root) holding the official test views and their ground truth composited on
the configured background, and scores every one of them. DTU has no such scene and fails
early.

### Unity project setup

Install Unity with Linux standalone build support. 
Set the usual path:

```bash
export UNITY=/path/to/Unity/Editor/Unity
```

`single_unity_eval.sh` applies the Vulkan compatibility patch automatically on
Linux. It also builds or reuses a Linux standalone Development Player at:

```text
outputs/unity_player/linux/MeshSuiteProfilePlayer.x86_64
```

### Export Unity assets only

```bash
./single_export_unity.sh 2dts mipnerf360/all cpu
./single_export_unity.sh triangle-splatting mipnerf360/garden cpu
```

Each scene writes:

```text
outputs/{method}/{dataset}/{scene}/unity_native/{method}.triasset/
```

Existing valid packages are skipped; use `--force` to re-export.

### Custom engine deployment

```bash
./single_unity_eval.sh 2dts mipnerf360/all 0 \
  --unity "$UNITY" \
  --unity-project unity
```

For all methods:

```bash
for method in 2dts triangle-splatting mesh-splatting diffsoup; do
  ./single_unity_eval.sh "$method" mipnerf360/all 0 \
    --unity "$UNITY" \
    --unity-project unity
done
```

The default setting is custom engine deployment (`--custom-engine`), which renders
through each method's method-aware Unity renderer; outputs are written under each
scene's `unity_custom_engine/` directory and summarized in:

```text
outputs/unity_reports/{method}/unity_custom_engine/unity_metrics_report.csv
outputs/unity_reports/{method}/unity_custom_engine/unity_metrics_report.json
outputs/unity_reports/{method}/unity_custom_engine/unity_metrics_report.md
```

Image metrics are also computed against each scene's de facto renderer outputs
(`--de-facto-render-subdir`, default `renders/test/renders`) and reported as
`de_facto_fidelity`.

The terminal summary reports:

```text
PSNR | SSIM | LPIPS | FPS | GPU P50 ms | GPU P95 ms | Alloc. MiB | Asset MiB
```

As in the v5 manuscript, `FPS` is end-to-end throughput including CPU processing
and GPU rendering: `1000 /` the median wall-clock engine frame time. GPU frame
latency (P50/P95) and renderer graphics allocations (`Alloc.`) are reported
separately.

Every Unity render target (held-out images, FPS timing, Player profiles and
videos) uses 2× MSAA, the Unity project's active quality level, for both
deployment settings. Each capture records the sample count Unity actually used
(`capture_settings.json`, and `msaa_samples` in each runtime profile), and the
scripts reject or re-capture results that were not rendered with 2× MSAA. Results
captured before this was made explicit had anti-aliased FPS targets but
non-anti-aliased images, and need to be re-captured.

### Standard engine deployment

Use `--standard-engine` to render a conventional opaque vertex-colored Unity Mesh;
outputs go to `unity_standard_engine/`:

```bash
for method in 2dts triangle-splatting mesh-splatting diffsoup; do
  ./single_unity_eval.sh "$method" mipnerf360/all 0 \
    --unity "$UNITY" \
    --unity-project unity \
    --standard-engine
done
```

Packages whose manifest does not declare a comparable standard engine deployment
appearance are rejected instead of producing misleading numbers.

DiffSoup has no vertex-color field, so `--standard-engine` first bakes its neural
textures (beyond the v5 manuscript). `tools/bake_diffsoup_standard_asset.py`
averages the decoder output of every lattice texel over the training views that see
it, then writes a separate package to `unity_baked/diffsoup.triasset`: the unchanged
triangle soup with a view-independent RGBA atlas. Unity draws it with a plain
textured alpha-cutout material (`MeshSuite/StandardTextureCutoutRaw`), so the MLP is
gone from the runtime, and `Asset MiB` covers the baked mesh only. Unlike the other
methods' opaque vertex-colored meshes, this keeps an alpha test: DiffSoup's 15k
triangles have too few vertices to carry its appearance. The bake needs CUDA and the
scene's COLMAP dataset, and `--force-export` re-bakes. A Player built before the
cutout shader existed needs `--force-player-build`.

### Mesh-oriented 2DTS (`2dts --opaque`)

`configs/2dts/` trains 2DTS as the volumetric radiance field its paper evaluates on
Mip-NeRF 360, Tanks & Temples and NeRF-Synthetic. The optional `--opaque` flag (method id
`2dts-opaque`) trains the same scenes with the same dataset protocol and primitive
budgets, but ends on solid triangles: gamma anneals from 1 to 50 over the last 10k of
30k iterations, opacity is binarised at 0.3, and the renderer sorts per pixel. Its
outputs stay separate, in `outputs/2dts-opaque/` and
`outputs/unity_reports/2dts-opaque/`.

The flag has no scene configs of its own. Each run uses the `configs/2dts` scene config
followed by `configs/base/2dts-opaque-<dataset>.yaml`, which adds the shared recipe in
`configs/base/2dts-opaque.yaml` and selects the dataset's opaque native preset.
Mip-NeRF 360, Tanks & Temples and NeRF-Synthetic have that overlay; DTU has none, since
`configs/2dts` already trains DTU opaque. `single_train.sh`,
`single_export_unity.sh`, `single_unity_eval.sh` and `single_unity_video.sh` accept
`--opaque`; the method id `2dts-opaque` names only the output tree and reports and is
not accepted as a method argument.

The schedule is the one the paper states for its mesh setting (arXiv v3, Sec. 4.3),
which it evaluates on NeRF-Synthetic and DTU only. Tanks & Temples takes its other
settings from the authors' mesh preset (`config/TanksAndTemples_VanillaTS.yaml`,
which itself runs 40k iterations). The authors publish no such preset or results for
Mip-NeRF 360, so `configs/2dts/native/mipnerf360-opaque.yaml` applies the same recipe
and takes `scale_pruning.radii_threshold` from their Mip-NeRF 360 preset.

For NeRF-Synthetic the authors run two stages: `config/NerfSynthetic_VanillaTS.yaml` for
30k iterations, then `config/NerfSynthetic_VanillaTS_mesh.yaml` from its 30k point
cloud to 60k (the 60k schedule of the paper's v1/v2). The single 30k run the paper
states takes precedence, so `configs/2dts/native/nerf_synthetic-opaque.yaml` applies the
Tanks & Temples recipe with the settings the NeRF-Synthetic presets change: their dataset
section, `max_grow_ratio`, scale pruning, and geometric losses from 3k with
`w_distortion` 0.05. It leaves out the second stage's settings and its `render_spp: 2`
(2x2 supersampling in training), since MeshSuite's de facto 2DTS renderer draws one
sample per pixel. The budget is the 2dts scene config's 1M triangles, while upstream
prunes its second stage to 41k–112k. On chair this recipe reaches 30.4 PSNR with about
19k triangles, against the paper's 34.16 with 84k faces (Table 10). Those numbers depend
on the paper's own renderer (its Table 9 drops the NeRF-Synthetic mean from 30.80 to
25.60 under Kaolin), so the gap is expected and left as is. This configuration is
beyond the v5 manuscript.

```bash
# De facto renderer: train, render the held-out views, score them.
bash single_train.sh 2dts mipnerf360/all 0 --opaque
bash single_train.sh 2dts tandt/all 0 --opaque
bash single_train.sh 2dts nerf_synthetic/all 0 --opaque

# Custom and standard engine deployment.
for dataset in mipnerf360 tandt nerf_synthetic; do
  ./single_unity_eval.sh 2dts "$dataset/all" 0 --opaque --unity "$UNITY" --unity-project unity \
    --report-name "unity_metrics_report_$dataset"
  ./single_unity_eval.sh 2dts "$dataset/all" 0 --opaque --unity "$UNITY" --unity-project unity \
    --report-name "unity_metrics_report_$dataset" --standard-engine
done

python3 tools/format_metrics.py --method 2dts-opaque --dataset mipnerf360/all
python3 tools/format_metrics.py --method 2dts-opaque --dataset tandt/all
python3 tools/format_metrics.py --method 2dts-opaque --dataset nerf_synthetic/all
```

The export applies the threshold the native renderer uses
(`adapter.render_params.ste_threshold`), so both deployments draw only the triangles
it draws. Standard engine deployment colors them with SH DC. Custom engine deployment
keeps the degree-3 SH and uses `MeshSuite/D2TSTerminalSolid`, a depth-buffered pass in
place of the sorted alpha blending used for volumetric 2DTS. The deployments' 2x MSAA
smooths hard triangle edges that the single-sample native renderer keeps, so custom
engine deployment can score slightly above the de facto renderer. A Player built
before that shader existed needs `--force-player-build`.

### Runtime profile details

By default, FPS and GPU latency are measured with the standalone Player:

```text
--profile-runtime player
--profile-runs 3
--profile-views 3
--profile-warmup 60
--profile-frames 180
--gpu-timing-min-fraction 0.5
```

This means each scene profiles three independent Player runs, three held-out
views per run, and 180 timed frames per view after warmup. Unity's
`FrameTimingManager` may not return GPU timing on every frame, so
`--gpu-timing-min-fraction` controls the minimum valid GPU sample coverage. A
profile with no valid GPU timing fails rather than falling back to CPU time.

For fast smoke tests:

```bash
./single_unity_eval.sh diffsoup mipnerf360/bicycle 0 \
  --unity "$UNITY" \
  --unity-project unity \
  --profile-runs 1 \
  --profile-views 1 \
  --profile-warmup 20 \
  --profile-frames 60 \
  --fps-warmup 3 \
  --fps-frames 12
```

If quality images and metrics already exist and only profiles need to be
rebuilt:

```bash
./single_unity_eval.sh diffsoup mipnerf360/all 0 \
  --unity "$UNITY" \
  --unity-project unity \
  --skip-capture \
  --skip-metrics \
  --force-profile
```

### Re-run every deployment metric

`batch_unity_eval.sh` re-evaluates both engine deployment settings for every method
and variant (`triangle-splatting`, `triangle-splatting-game-engine`
(`triangle-splatting --train_game_engine`), `2dts`,
`2dts-opaque` (`2dts --opaque`), `diffsoup` with its baked Std. package,
`mesh-splatting`) on
Mip-NeRF 360, Tanks and Temples and NeRF-Synthetic, and merges the results into one
text file:

```bash
bash batch_unity_eval.sh 0 --unity "$UNITY" --unity-project unity --display "$DISPLAY"
# after a failure, finish the remaining scenes with the same options:
bash batch_unity_eval.sh 0 --unity "$UNITY" --unity-project unity --display "$DISPLAY" --resume
```

Each scene runs `single_unity_eval.sh` with every stage forced (export or bake,
capture, Player profile, metrics), and the standalone Player is rebuilt once, so no
earlier number survives. A failed scene is retried (`--retries`, default 2) and the run
continues. One report per method x setting x dataset holds only the scenes this
campaign completed. `tools/summarize_deployment_metrics.py` then writes
`outputs/deployment_eval/deployment_metrics_summary.txt`: an overview of dataset means,
then a per-scene table for each group, with missing or incomplete groups marked.
A companion `.csv` holds the same rows at full precision. `--methods`,
`--datasets` and `--settings` select a subset, `--dry-run` resolves every command
without running Unity, `--summary-only` rebuilds the text file, and options after
`--` go to every `single_unity_eval.sh` call.

### Resolution and dataset rules

Every method, 2DTS included, reads the stored `images_2`/`images_4` folders the
scene config names (`image_dir: images` with `resolution: 2/4/8` resolves to
`images_2/4/8`). A missing prepared folder or image fails instead of being
downsampled at load time.

The historical 2DTS input (offline PIL bilinear resizing with integer-floor
dimensions, saved as lossless PNG) is an opt-in protocol: set
`dataset.image_preprocessing: legacy_bilinear_floor` with `image_dir: images` and
`resolution: 2/4/8`, then prepare the caches in the 2DTS runtime environment:

```bash
python -m tools.prepare_2dts_images --config path/to/scene-configs --workers 4
```

Its pixels differ from the stored `images_N` JPEGs every other method reads (the
same 2DTS bicycle checkpoint scores 26.25 dB on it and 25.15 dB on `images_4`), so
its results are not comparable with the shared protocol. Caches live in
`.cache/msbench/images/` (override with `MSBENCH_IMAGE_CACHE_DIR`, using the same
value for preparation and all runs). The manifest records the protocol, Pillow
version, source hashes, pixel hashes, dimensions and source file metadata.
Training/evaluation reject missing caches and changed source files. Explicit
cache directories can also be selected with `image_dir` and `resolution: 1`. DTU
and synthetic preprocessing remain as configured.

COLMAP loaders set the effective resize factor to `1` for prepared images.
Stored pixel dimensions determine camera size and scaled intrinsics.
Structured training records the effective source in `dataset_source.json` beside
the original `config.yaml`. Explicit custom image folders and target-width
controls retain their existing semantics.
The Unity camera replay currently requires COLMAP `sparse/0/cameras.bin` and
`sparse/0/images.bin`.

## Config Inheritance

Experiment configs use `_base_` inheritance. Common defaults live in
`configs/base/_base_.yaml`, dataset-level defaults live in files such as
`configs/base/mipnerf360.yaml`, method-level shared settings live in files such
as `configs/base/triangle-splatting.yaml`, and method/dataset/scene configs
only keep the fields that are unique to that run.

Scene configs are grouped by method and dataset:

```text
configs/triangle-splatting/
├── dtu/scan24.yaml
├── mipnerf360/bicycle.yaml
└── tandt/truck.yaml
configs/mesh-splatting/
├── dtu/scan24.yaml
├── mipnerf360/bicycle.yaml
└── tandt/truck.yaml
configs/2dts/
├── dtu/scan24.yaml
├── mipnerf360/bicycle.yaml
├── native/dtu.yaml
└── tandt/truck.yaml
configs/exmesh/
├── dtu/scan24.yaml
└── nerf_synthetic/lego.yaml
```

For example, `configs/triangle-splatting/mipnerf360/bicycle.yaml` inherits the
MipNeRF360 dataset defaults plus Triangle Splatting defaults, then only sets the
scene and scene-specific options:

```yaml
_base_:
  - ../../base/mipnerf360.yaml
  - ../../base/triangle-splatting.yaml

experiment:
  name: triangle_splatting_bicycle

dataset:
  scene: bicycle
  image_dir: images_4
  resolution: 1

trainer:
  outdoor: true
```

`dataset.root` and `dataset.scene` are resolved together, so a dataset base can
declare `root: data/mipnerf360` while a scene config declares `scene: bicycle`;
the effective dataset path becomes `data/mipnerf360/bicycle`. String templates
such as `outputs/{method}/{dataset}/{scene}` are resolved after all inherited
configs are merged.

To run a whole dataset, use `single_train.sh` with a dataset target ending in
`/all`. It expands the configured scene list, trains each scene, renders/evals,
exports videos, and prints a summary table:

```bash
bash single_train.sh triangle-splatting mipnerf360/all 0
bash single_train.sh triangle-splatting tandt/all 0
bash single_train.sh triangle-splatting dtu/all 0
bash single_train.sh mesh-splatting mipnerf360/all 0
bash single_train.sh mesh-splatting tandt/all 0
bash single_train.sh mesh-splatting dtu/all 0
bash single_train.sh 2dts mipnerf360/all 0
bash single_train.sh 2dts tandt/all 0
bash single_train.sh 2dts dtu/all 0
```

Use `all` as the target to run all built-in datasets in one pass.

> **DTU Evaluation Modes**:
> DTU training and evaluation defaults to masked foreground evaluation composited on a white background (`--dtu-foreground` or alias `--dtu-masked`). A config that sets `dataset.background` explicitly keeps that background in foreground mode. To train and evaluate on unmasked full images with natural backgrounds (Mode A), pass `--dtu-full` or alias `--dtu-unmasked`:
> ```bash
> # Unmasked full-image evaluation (Mode A)
> bash single_train.sh triangle-splatting dtu/all 0 --dtu-unmasked
>
> # Masked white-background evaluation (Mode B, default)
> bash single_train.sh triangle-splatting dtu/all 0 --dtu-masked
> ```

Triangle Splatting configs inherit scene-specific triangle count caps from
`configs/base/triangle-splatting-caps.yaml` through
`configs/base/triangle-splatting.yaml`. During config finalization, the cap for
`dataset.scene` is written to `trainer.max_shapes` unless the scene config
explicitly overrides it.

MeshSplatting configs live under `configs/mesh-splatting/` for MipNeRF360,
Tanks&Temples, and DTU. They use the bundled MeshSplatting renderer and a
method-specific trainer that preserves shared-vertex topology updates. DTU
configs follow the upstream DTU settings, including restricted Delaunay at
iteration 11000; that stage requires the MeshSplatting `effrdel` dependency to
be installed in the active environment.

## Unified Output Layout

Every method now writes to one consistent per-scene directory so training
checkpoints, rendered images, exported geometry, videos, and logs always live
in the same place regardless of backend.

### Target layout

```text
outputs/{method}/{dataset}/{scene}/
├── ckpt/                 # all training checkpoints (one folder, every method)
│   ├── 30000.ckpt                       # 2DTS
│   ├── point_cloud/{N}.ply              # 2DTS exported point clouds
│   ├── point_cloud/iteration_{N}/*.pt   # triangle-/mesh-splatting
│   └── final_params.pt                  # DiffSoup
├── renders/              # rendered images + matching ground truth, by split
│   ├── train/
│   │   ├── renders/      # predicted RGB
│   │   ├── gt/           # dataset ground truth
│   │   ├── manifest.json
│   │   └── metrics.json
│   └── test/
│       ├── renders/
│       ├── gt/
│       ├── manifest.json
│       └── metrics.json
├── mesh/                 # exported geometry used for mesh/DTU metrics
│   ├── fuse_post.ply                    # TSDF mesh (mip360/tandt/dtu)
│   └── {N}_pcd.ply                      # 2DTS DTU point-cloud export
├── video/                # trajectory video
│   ├── render_traj.mp4
│   └── frames/
├── logs/                 # every log: pipeline *.log, training stdout,
│                         # tensorboard events, cfg dumps, loss curves
├── config.yaml           # resolved config snapshot
├── metrics.json          # aggregated image metrics (test split)
├── mesh_metrics.json     # DTU/mesh geometry metrics
├── profile.json          # renderer profiling report (msbench profile)
├── stats.json            # model statistics (msbench inspect)
└── train_stats.json      # training time / peak GPU memory
```

### Adapter checkpoint paths

The `adapter.checkpoint` template in each method base config points at `ckpt/`:

| Method             | `adapter.checkpoint`                                            |
| ------------------ | -------------------------------------------------------------- |
| 2DTS               | `outputs/{method}/{dataset}/{scene}/ckpt`                      |
| Triangle Splatting | `outputs/{method}/{dataset}/{scene}/ckpt/point_cloud/iteration_{max_steps}` |
| MeshSplatting      | `outputs/{method}/{dataset}/{scene}/ckpt/point_cloud/iteration_{max_steps}` |
| DiffSoup           | `outputs/{method}/{dataset}/{scene}/ckpt/final_params.pt`     |
| ExMesh             | `outputs/{method}/{dataset}/{scene}/ckpt/final.pt`             |
| T²-Splat           | `outputs/{method}/{dataset}/{scene}/ckpt/point_cloud/iteration_{max_steps}` |


## Project Structure


```
MeshSuite/
├── msbench/
│   ├── core/           # Config, datasets, cameras, metrics, mesh eval, profiling, viewer
│   ├── primitives/     # Triangle primitive types (independent, mesh, convex)
│   ├── renderers/      # Adapter wrappers for each method
│   ├── trainers/       # Training loop, per-method trainers, losses, hooks
│   ├── cli/            # Typer-based CLI commands
│   ├── vendor/         # Bundled upstream method code and CUDA extensions
│   └── tests/          # pytest test suite
├── configs/            # YAML experiment configs (base/ + <method>/<dataset>/<scene>.yaml)
├── tools/              # Metrics formatting, Unity export/eval, preprocessing scripts
├── unity/              # Unity project used for engine-side evaluation
├── single_*.sh         # Scene/dataset pipeline wrappers (train, export, Unity eval)
└── pyproject.toml
```


## Citation

If you find this benchmark, codebase, or results helpful in your research, please cite our paper:

```bibtex
@misc{zhang2026meshsuiteunifiedbenchmarktriangle,
      title={MeshSuite: A Unified Benchmark for Triangle- and Mesh-Based Neural Rendering}, 
      author={Kaixuan Zhang and Minxian Li and Mingwu Ren and Xiatian Zhu},
      year={2026},
      eprint={2609.01306},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2609.01306}, 
}
```

## License

This project is licensed under the [MIT License](LICENSE). Third-party submodules and vendor components under `msbench/vendor/` and `submodules/` are subject to their respective original licenses.
